"""Join target-free candidate-pose LLR scores to pose targets for one audit.

The scorer writes no targets and cannot promote its raw visual LLR.  This
separate program is the sole target reader: it compares a visual score to an
identical support-descriptor-permutation control, reports pose ranking and
selection changes, and leaves PnP integration forbidden regardless of the
outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


# Keep the documented direct-script audit invocation independent of a caller
# pre-populating PYTHONPATH, matching the training and scoring entry points.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_pose_evidence import (
    TARGET_FORMAT,
    _baseline_rows,
    _paired,
    _per_query_rows,
    _split_summary,
    _validate_alpha_zero,
    _write_csv,
)
from feature_extract.tools.vfm.audit_v5_dynamic_absolute_context_pose_evidence import (
    _row_keys,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_SCORE_FORMAT,
    validate_target_free_pose_llr_score_metadata,
)


_EPSILON = 1e-12
_GROUPED_TARGET_FORMAT = "grouped_pose_hypothesis_targets_v1"
_SUPPORTED_TARGET_FORMATS = {TARGET_FORMAT, _GROUPED_TARGET_FORMAT}
_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "source_chosen_for_optional_pose",
    "baseline_score_top1",
    "baseline_selection_scores",
    "pose_log_likelihood_ratios",
    "point_log_likelihood_ratios",
    "point_effective_candidate_counts",
    "point_effective_view_masses",
    "source_log_likelihood_means",
    "source_effective_point_counts",
)
_FROZEN_ARRAY_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "source_chosen_for_optional_pose",
    "baseline_score_top1",
    "baseline_selection_scores",
    "verification_source_point_ids",
    "verification_point_sources",
    "verification_source_detector_rows",
    "verification_xy",
    "candidate_track_ids",
    "candidate_probabilities",
    "null_probabilities",
    "candidate_view_weights",
    "candidate_support_image_ids",
    "source_names",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-score-artifacts", required=True)
    parser.add_argument("--control-score-artifacts", required=True)
    parser.add_argument("--target-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("candidate pose-LLR audit paths must be non-empty and unique")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _strict_contract(metadata: Mapping[str, object], *, variant: str) -> Mapping[str, object]:
    strict = metadata.get("strict_candidate_pose_llr_contract")
    required = {
        "heldout_query_rows": True,
        "formal_p1_mixed_multiscale_points": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "fixed_support_view_count": 2,
        "explicit_null": True,
        "candidate_projection_is_only_pose_dependent_encoder_input": True,
        "candidate_pose_matrix_excluded_from_encoder": True,
        "residual_and_target_excluded_from_encoder": True,
        "no_pnp": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if (
        not isinstance(strict, Mapping)
        or any(strict.get(key) != value for key, value in required.items())
        or bool(strict.get("support_descriptor_permutation_control"))
        != (str(variant) == "support_descriptor_permutation_control")
    ):
        raise ValueError("candidate pose-LLR score has an invalid strict contract")
    # v2 records the actual control mechanism instead of implying that an RGB
    # crop tensor was merely permuted.  If that field is present, it must agree
    # with the historic paired-audit variant and prove that geometry remained
    # fixed.  Older score artifacts remain readable for historical audits.
    if "support_image_appearance_derangement_control" in strict and (
        strict.get("support_image_appearance_derangement_control")
        != (str(variant) == "support_descriptor_permutation_control")
        or strict.get("appearance_control_geometry_fixed") is not True
    ):
        raise ValueError("candidate pose-LLR score lacks a geometry-fixed image control")
    return strict


def _load_score(
    path: Path, *, expected_variant: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    required = set(_ROW_FIELDS) | set(_FROZEN_ARRAY_FIELDS)
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: candidate pose-LLR score is incomplete ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in required}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: candidate pose-LLR metadata is invalid")
    validate_target_free_pose_llr_score_metadata(metadata)
    if (
        metadata.get("format") != CANDIDATE_POSE_LLR_SCORE_FORMAT
        or str(metadata.get("evidence_variant")) != str(expected_variant)
    ):
        raise ValueError(f"{path}: candidate pose-LLR score has the wrong evidence variant")
    _strict_contract(metadata, variant=str(expected_variant))
    count = len(np.asarray(arrays["query_ids"]))
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in _ROW_FIELDS):
        raise ValueError(f"{path}: candidate pose-LLR row arrays are inconsistent")
    point_logs = np.asarray(arrays["point_log_likelihood_ratios"], dtype=np.float64)
    point_counts = np.asarray(arrays["point_effective_candidate_counts"], dtype=np.int64)
    point_masses = np.asarray(arrays["point_effective_view_masses"], dtype=np.float64)
    source_names = np.asarray(arrays["source_names"]).astype(str).reshape(-1)
    source_logs = np.asarray(arrays["source_log_likelihood_means"], dtype=np.float64)
    source_counts = np.asarray(arrays["source_effective_point_counts"], dtype=np.int64)
    if (
        point_logs.ndim != 2
        or point_counts.shape != point_logs.shape
        or point_masses.shape != point_logs.shape
        or source_names.size == 0
        or len(set(source_names.tolist())) != len(source_names)
        or source_logs.shape != (count, len(source_names))
        or source_counts.shape != source_logs.shape
        or not np.isfinite(point_logs).all()
        or not np.isfinite(point_masses).all()
        or not np.isfinite(source_logs).all()
        or np.any(point_counts < 0)
        or np.any(point_masses < 0.0)
        or np.any(source_counts < 0)
    ):
        raise ValueError(f"{path}: candidate pose-LLR evidence arrays are invalid")
    if (
        np.asarray(arrays["verification_source_point_ids"]).shape != (point_logs.shape[1],)
        or np.asarray(arrays["verification_point_sources"]).shape != (point_logs.shape[1],)
        or np.asarray(arrays["verification_source_detector_rows"]).shape != (point_logs.shape[1],)
        or np.asarray(arrays["verification_xy"]).shape != (point_logs.shape[1], 2)
    ):
        raise ValueError(f"{path}: candidate pose-LLR static point layout is invalid")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: candidate pose-LLR score repeats frozen rows")
    return arrays, metadata


def _normalized_strict_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    strict = _strict_contract(metadata, variant=str(metadata.get("evidence_variant")))
    output = dict(strict)
    output.pop("support_descriptor_permutation_control", None)
    output.pop("support_image_appearance_derangement_control", None)
    return output


def _assert_visual_control_pair(
    *,
    visual_arrays: Mapping[str, np.ndarray],
    visual_metadata: Mapping[str, object],
    control_arrays: Mapping[str, np.ndarray],
    control_metadata: Mapping[str, object],
) -> None:
    """Require geometry and all fixed denominators to be identical across control."""

    _strict_contract(visual_metadata, variant="visual")
    _strict_contract(
        control_metadata, variant="support_descriptor_permutation_control"
    )
    if (
        str(visual_metadata.get("frozen_query_evidence_sha256", ""))
        != str(control_metadata.get("frozen_query_evidence_sha256", ""))
        or _normalized_strict_contract(visual_metadata)
        != _normalized_strict_contract(control_metadata)
        or visual_metadata.get("verification_point_selection")
        != control_metadata.get("verification_point_selection")
    ):
        raise ValueError("candidate pose-LLR visual/control metadata differs beyond appearance")
    for field in _FROZEN_ARRAY_FIELDS:
        if not np.array_equal(np.asarray(visual_arrays[field]), np.asarray(control_arrays[field])):
            raise ValueError(f"candidate pose-LLR frozen field differs: {field}")
    for field in ("point_effective_candidate_counts", "point_effective_view_masses"):
        if not np.array_equal(np.asarray(visual_arrays[field]), np.asarray(control_arrays[field])):
            raise ValueError(f"candidate pose-LLR geometry-only field differs: {field}")


def _merge_pairs(
    *,
    visual_paths: Sequence[Path],
    control_paths: Sequence[Path],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[tuple[str, str, str, int], ...],
]:
    visual_loaded = [_load_score(path, expected_variant="visual") for path in visual_paths]
    control_loaded = [
        _load_score(path, expected_variant="support_descriptor_permutation_control")
        for path in control_paths
    ]
    visual_by_query = {str(metadata.get("query_id")): (arrays, metadata) for arrays, metadata in visual_loaded}
    control_by_query = {str(metadata.get("query_id")): (arrays, metadata) for arrays, metadata in control_loaded}
    if (
        not visual_by_query
        or set(visual_by_query) != set(control_by_query)
        or len(visual_by_query) != len(visual_loaded)
        or len(control_by_query) != len(control_loaded)
    ):
        raise ValueError("candidate pose-LLR visual/control shards do not cover identical queries")
    visual_parts: dict[str, list[np.ndarray]] = {field: [] for field in _ROW_FIELDS}
    control_parts: dict[str, list[np.ndarray]] = {field: [] for field in _ROW_FIELDS}
    visual_metadata: list[dict[str, object]] = []
    control_metadata: list[dict[str, object]] = []
    reference_sources: np.ndarray | None = None
    for query_id in sorted(visual_by_query):
        visual_arrays, visual_item = visual_by_query[query_id]
        control_arrays, control_item = control_by_query[query_id]
        _assert_visual_control_pair(
            visual_arrays=visual_arrays,
            visual_metadata=visual_item,
            control_arrays=control_arrays,
            control_metadata=control_item,
        )
        source_names = np.asarray(visual_arrays["source_names"]).astype(str)
        if reference_sources is None:
            reference_sources = source_names
        elif not np.array_equal(source_names, reference_sources):
            raise ValueError("candidate pose-LLR shards expose different source names")
        for field in _ROW_FIELDS:
            visual_parts[field].append(np.asarray(visual_arrays[field]))
            control_parts[field].append(np.asarray(control_arrays[field]))
        visual_metadata.append(visual_item)
        control_metadata.append(control_item)
    visual = {field: np.concatenate(parts, axis=0) for field, parts in visual_parts.items()}
    control = {field: np.concatenate(parts, axis=0) for field, parts in control_parts.items()}
    assert reference_sources is not None
    visual["source_names"] = reference_sources
    control["source_names"] = reference_sources
    keys = _row_keys(visual)
    if len(keys) != len(set(keys)) or keys != _row_keys(control):
        raise ValueError("candidate pose-LLR merged visual/control row identities differ")
    return visual, control, visual_metadata, control_metadata, keys


def _paired_rank(
    visual_rows: Sequence[Mapping[str, object]], control_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    visual = {key(row): row for row in visual_rows}
    control = {key(row): row for row in control_rows}
    if not visual or set(visual) != set(control):
        raise ValueError("candidate pose-LLR visual/control rows are unpaired")
    deltas = []
    for item in sorted(visual):
        left = visual[item].get("best_10cm_rank")
        right = control[item].get("best_10cm_rank")
        if left is not None and right is not None:
            deltas.append(float(left) - float(right))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "best_10cm_rank_pair_count": int(len(values)),
        "best_10cm_rank_wins": int(np.count_nonzero(values < -_EPSILON)),
        "best_10cm_rank_losses": int(np.count_nonzero(values > _EPSILON)),
        "best_10cm_rank_ties": int(np.count_nonzero(np.abs(values) <= _EPSILON)),
        "median_best_10cm_rank_delta": (
            None if len(values) == 0 else float(np.median(values))
        ),
    }


def _primary_gate(
    *,
    visual: Mapping[str, object],
    control: Mapping[str, object],
    paired_rank: Mapping[str, object],
) -> dict[str, bool]:
    """Require independent visual evidence to move correct hypotheses toward top-20."""

    visual_median = visual.get("median_best_10cm_rank")
    control_median = control.get("median_best_10cm_rank")
    visual_p90 = visual.get("p90_best_10cm_rank")
    control_p90 = control.get("p90_best_10cm_rank")
    return {
        "visual_median_best_10cm_rank_within_top20": visual_median is not None
        and float(visual_median) <= 20.0,
        "visual_median_best_10cm_rank_below_control": visual_median is not None
        and control_median is not None
        and float(visual_median) < float(control_median),
        "visual_p90_best_10cm_rank_not_worse": visual_p90 is not None
        and control_p90 is not None
        and float(visual_p90) <= float(control_p90),
        "visual_p90_translation_not_worse": float(visual["p90_selected_translation_cm"])
        <= float(control["p90_selected_translation_cm"]),
        "visual_catastrophic_tail_not_worse": int(visual["catastrophic_1m_count"])
        <= int(control["catastrophic_1m_count"]),
        "paired_best_10cm_rank_wins_exceed_losses": int(
            paired_rank["best_10cm_rank_wins"]
        )
        > int(paired_rank["best_10cm_rank_losses"]),
    }


def _source_summary(values: Mapping[str, np.ndarray]) -> dict[str, object]:
    names = np.asarray(values["source_names"]).astype(str)
    logs = np.asarray(values["source_log_likelihood_means"], dtype=np.float64)
    counts = np.asarray(values["source_effective_point_counts"], dtype=np.float64)
    return {
        str(name): {
            "mean_log_likelihood_ratio": float(np.mean(logs[:, index])),
            "mean_effective_point_count": float(np.mean(counts[:, index])),
        }
        for index, name in enumerate(names.tolist())
    }


def _load_targets(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load either the S0 score-target join or the grouped-inference target join."""

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
            raise ValueError(f"{path}: candidate pose-LLR target is incomplete ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") not in _SUPPORTED_TARGET_FORMATS
        or metadata.get("contains_target_fields") is not True
    ):
        raise ValueError("candidate pose-LLR target is not a supported post-inference join")
    if (
        metadata.get("format") == _GROUPED_TARGET_FORMAT
        and metadata.get("targets_joined_after_inference") is not True
    ):
        raise ValueError("candidate pose-LLR grouped target was not joined after inference")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(value).shape[0] != count for value in arrays.values()):
        raise ValueError("candidate pose-LLR target arrays are not row aligned")
    translation = np.asarray(arrays["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(arrays["rotation_errors_deg"], dtype=np.float64)
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
        or len(_row_keys(arrays)) != len(set(_row_keys(arrays)))
    ):
        raise ValueError("candidate pose-LLR target values are invalid")
    return arrays, metadata


def _validate_target_lineage(
    *, score_metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    target_format = str(target_metadata.get("format", ""))
    if target_format not in _SUPPORTED_TARGET_FORMATS:
        raise ValueError("candidate pose-LLR target artifact has the wrong format")
    hypotheses = set(
        str(value)
        for value in target_metadata.get(
            "inference_artifact_sha256"
            if target_format == _GROUPED_TARGET_FORMAT
            else "hypothesis_artifact_sha256",
            [],
        )
    )
    baselines = set(str(value) for value in target_metadata.get("score_artifact_sha256", []))
    if not hypotheses:
        raise ValueError("candidate pose-LLR target has no inference lineage")
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("candidate pose-LLR score has no frozen input manifest")
        if (
            str(inputs.get("hypothesis_artifact", {}).get("sha256", "")) not in hypotheses
            or (
                bool(baselines)
                and str(inputs.get("baseline_score_artifact", {}).get("sha256", ""))
                not in baselines
            )
        ):
            raise ValueError("candidate pose-LLR score/target lineage differs from frozen S0")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing audit: {summary_path}")
    visual_paths = _paths(args.visual_score_artifacts)
    control_paths = _paths(args.control_score_artifacts)
    visual, control, visual_metadata, control_metadata, keys = _merge_pairs(
        visual_paths=visual_paths, control_paths=control_paths
    )
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    _validate_target_lineage(
        score_metadata=[*visual_metadata, *control_metadata], target_metadata=target_metadata
    )
    target_positions = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        target_rows = np.asarray([target_positions[key] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("candidate pose-LLR score row is absent from the target artifact") from error
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline_scores = np.asarray(visual["baseline_selection_scores"], dtype=np.float64)
    source_top1 = np.asarray(visual["baseline_score_top1"], dtype=bool)
    if (
        not np.isfinite(baseline_scores).all()
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or translation.shape != baseline_scores.shape
    ):
        raise ValueError("candidate pose-LLR target values are misaligned")
    tie_orders = np.arange(len(keys), dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=keys,
        baseline_scores=baseline_scores,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
    )
    _validate_alpha_zero(
        source_top1=source_top1, keys=keys, baseline_rows=baseline_rows
    )
    visual_rows = _per_query_rows(
        keys=keys,
        scores=np.asarray(visual["pose_log_likelihood_ratios"], dtype=np.float64),
        baseline_scores=baseline_scores,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
        family="candidate_pose_llr_visual",
        score_mode="standalone_raw_llr_diagnostic_only",
        alpha=None,
    )
    control_rows = _per_query_rows(
        keys=keys,
        scores=np.asarray(control["pose_log_likelihood_ratios"], dtype=np.float64),
        baseline_scores=baseline_scores,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
        family="candidate_pose_llr_support_descriptor_permutation_control",
        score_mode="standalone_raw_llr_diagnostic_only",
        alpha=None,
    )
    visual_summary = _split_summary(visual_rows)
    control_summary = _split_summary(control_rows)
    paired_rank = _paired_rank(visual_rows, control_rows)
    split_gates = {
        split: _primary_gate(
            visual=visual_summary[split],
            control=control_summary[split],
            paired_rank=_paired_rank(
                [row for row in visual_rows if str(row["split_name"]) == split],
                [row for row in control_rows if str(row["split_name"]) == split],
            ),
        )
        for split in sorted(set(visual_summary).intersection(control_summary))
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", [*baseline_rows, *visual_rows, *control_rows])
    summary: dict[str, Any] = {
        "stage": "candidate_pose_llr_target_audit",
        "format": "candidate_pose_llr_target_audit_v1",
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "visual_and_descriptor_permutation_control_paired": True,
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
            "paired_vs_s0": _paired(baseline_rows, visual_rows),
            "source_evidence": _source_summary(visual),
        },
        "support_descriptor_permutation_control": {
            "splits": control_summary,
            "paired_vs_s0": _paired(baseline_rows, control_rows),
            "source_evidence": _source_summary(control),
        },
        "visual_vs_control": {
            "paired_best_10cm_rank": paired_rank,
            "split_primary_gate": split_gates,
            "all_split_primary_gates_pass": bool(split_gates)
            and all(all(gate.values()) for gate in split_gates.values()),
        },
        "inputs": {
            "visual_score_artifacts": [str(path) for path in visual_paths],
            "visual_score_artifact_sha256": [file_sha256_short(path) for path in visual_paths],
            "control_score_artifacts": [str(path) for path in control_paths],
            "control_score_artifact_sha256": [file_sha256_short(path) for path in control_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "score_contract_sha256": _canonical_hash(
                [_normalized_strict_contract(item) for item in visual_metadata]
            ),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
