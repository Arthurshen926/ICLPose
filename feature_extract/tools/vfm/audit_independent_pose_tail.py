"""Audit frozen independent-pose score failures after a target-only join.

The independent scorer deliberately does not load poses or target errors.  This
tool is the corresponding post-hoc diagnostic: it joins immutable score rows
to a separately produced target artifact and explains *what was observable* at
the selected and oracle hypotheses.  It never changes scores, thresholds, or
pose selection.

In particular, an aggregate score file cannot attribute a failure to an RGB
mode, a support view, or a spatial block unless it carries a matching sidecar.
The report states that limitation instead of inventing an explanation.  This is
important for the current no-spatial S18 score run, where a high-confidence
wrong pose must not be described as an RGB-spatial failure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


SCORE_FORMAT = "independent_landmark_hypothesis_scores_v1"
TARGET_FORMAT = "independent_landmark_hypothesis_score_targets_v1"
REPORT_FORMAT = "independent_pose_tail_audit_v1"

_KEY_FIELDS = ("query_ids", "split_names", "evaluation_labels", "hypothesis_indices")
_SCORE_FIELDS = (
    *_KEY_FIELDS,
    "independent_selection_scores",
    "independent_log_likelihood_means",
    "independent_log_likelihood_medians",
    "independent_log_likelihood_worst_quartile_means",
    "independent_log_likelihood_lcb95s",
    "independent_spatial_median_of_means_2x2",
    "independent_effective_point_counts",
    "independent_evidence_coverages",
    "verification_point_counts",
    "candidate_spatial_materialized_verification_point_counts",
    "candidate_spatial_materialized_candidate_view_counts",
)
_TARGET_FIELDS = (*_KEY_FIELDS, "translation_errors_m", "rotation_errors_deg")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score_artifacts",
        required=True,
        help="comma-separated target-free score NPZ shards",
    )
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--tail_threshold_m", type=float, default=1.0)
    parser.add_argument("--good_pose_threshold_m", type=float, default=0.10)
    parser.add_argument("--low_coverage_threshold", type=float, default=0.50)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_npz(path: Path, *, expected_format: str, fields: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in fields if name not in payload.files]
        if missing:
            raise ValueError(f"{path} lacks required field {missing[0]!r}")
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path} lacks metadata_json")
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {name: np.asarray(payload[name]) for name in fields}
    if not isinstance(metadata, dict) or metadata.get("format") != expected_format:
        raise ValueError(f"{path} has an incompatible artifact format")
    row_count = len(arrays[_KEY_FIELDS[0]])
    if row_count == 0 or any(len(arrays[name]) != row_count for name in fields):
        raise ValueError(f"{path} has inconsistent row-aligned arrays")
    return arrays, metadata


def _key_rows(arrays: Mapping[str, np.ndarray]) -> list[tuple[str, str, str, int]]:
    return [
        (str(query), str(split), str(label), int(index))
        for query, split, label, index in zip(
            arrays["query_ids"].astype(str),
            arrays["split_names"].astype(str),
            arrays["evaluation_labels"].astype(str),
            arrays["hypothesis_indices"].astype(np.int64),
        )
    ]


def _merge_score_shards(paths: Sequence[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("tail audit requires at least one score artifact")
    parts: list[dict[str, np.ndarray]] = []
    metadata_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for path in paths:
        arrays, metadata = _load_npz(path, expected_format=SCORE_FORMAT, fields=_SCORE_FIELDS)
        if metadata.get("contains_target_fields") is not False or metadata.get(
            "pose_or_ground_truth_used_for_scoring"
        ) is not False:
            raise ValueError("tail audit accepts only target-free independent score artifacts")
        keys = _key_rows(arrays)
        duplicate = next((key for key in keys if key in seen), None)
        if duplicate is not None:
            raise ValueError(f"score artifacts contain duplicate hypothesis key {duplicate!r}")
        seen.update(keys)
        parts.append(arrays)
        metadata_rows.append(metadata)
    return {name: np.concatenate([part[name] for part in parts], axis=0) for name in _SCORE_FIELDS}, metadata_rows


def _join_targets(
    score: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    target_keys = _key_rows(target)
    target_index = {key: index for index, key in enumerate(target_keys)}
    if len(target_index) != len(target_keys):
        raise ValueError("target artifact contains duplicate hypothesis keys")
    lookup = np.asarray(
        [target_index.get(key, -1) for key in _key_rows(score)], dtype=np.int64
    )
    if np.any(lookup < 0):
        raise ValueError("target artifact does not cover every frozen score row")
    return (
        np.asarray(target["translation_errors_m"], dtype=np.float64)[lookup],
        np.asarray(target["rotation_errors_deg"], dtype=np.float64)[lookup],
    )


def _rank_descending(values: np.ndarray) -> np.ndarray:
    """Stable descending ranks, with equal values receiving the best rank."""

    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty((len(order),), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    first = np.maximum.accumulate(np.where(starts, np.arange(len(order)), 0)) + 1
    ranks[order] = first
    return ranks


def _score_margin(scores: np.ndarray, selected_index: int) -> float:
    if len(scores) < 2:
        return float("nan")
    selected = float(scores[selected_index])
    alternatives = np.delete(np.asarray(scores, dtype=np.float64), selected_index)
    return float(selected - np.max(alternatives))


def _mode_status(score_metadata: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    materialized_points = np.asarray(
        arrays["candidate_spatial_materialized_verification_point_counts"], dtype=np.int64
    )
    materialized_views = np.asarray(
        arrays["candidate_spatial_materialized_candidate_view_counts"], dtype=np.int64
    )
    input_rows = [metadata.get("inputs", {}) for metadata in score_metadata]
    declared_paths = [
        path
        for inputs in input_rows
        if isinstance(inputs, Mapping)
        for path in inputs.get("candidate_spatial_likelihood", [])
    ]
    return {
        "candidate_spatial_artifact_declared": bool(declared_paths),
        "candidate_spatial_artifact_paths": [str(path) for path in declared_paths],
        "materialized_verification_point_count": int(np.max(materialized_points, initial=0)),
        "materialized_candidate_view_count": int(np.max(materialized_views, initial=0)),
        "spatial_modes_available_for_this_score": bool(
            np.any(materialized_points > 0) and np.any(materialized_views > 0)
        ),
        "per_point_or_block_sidecar_available": False,
        "attribution_limit": (
            "aggregate_only_no_per_point_or_block_sidecar_v1"
        ),
    }


def audit_independent_pose_tail(
    *,
    score_artifacts: Sequence[Path],
    target_artifact: Path,
    tail_threshold_m: float = 1.0,
    good_pose_threshold_m: float = 0.10,
    low_coverage_threshold: float = 0.50,
    top_n: int = 20,
) -> dict[str, Any]:
    """Return a target-only tail report without mutating score selection."""

    if (
        float(tail_threshold_m) <= 0.0
        or float(good_pose_threshold_m) <= 0.0
        or not 0.0 <= float(low_coverage_threshold) <= 1.0
        or int(top_n) <= 0
    ):
        raise ValueError("tail audit thresholds are invalid")
    score, score_metadata = _merge_score_shards([Path(path) for path in score_artifacts])
    target, target_metadata = _load_npz(
        Path(target_artifact), expected_format=TARGET_FORMAT, fields=_TARGET_FIELDS
    )
    if target_metadata.get("contains_target_fields") is not True:
        raise ValueError("tail audit target artifact must be target-bearing")
    translation, rotation = _join_targets(score, target)
    mode_status = _mode_status(score_metadata, score)
    keys = _key_rows(score)
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row, key in enumerate(keys):
        grouped.setdefault(key[:3], []).append(row)

    rows: list[dict[str, Any]] = []
    score_names = {
        "mean": "independent_log_likelihood_means",
        "median": "independent_log_likelihood_medians",
        "worst_quartile": "independent_log_likelihood_worst_quartile_means",
        "lcb95": "independent_log_likelihood_lcb95s",
        "spatial_median_of_means": "independent_spatial_median_of_means_2x2",
    }
    for (query_id, split_name, label), indices in sorted(grouped.items()):
        ordered = np.asarray(indices, dtype=np.int64)
        selection = np.asarray(score["independent_selection_scores"], dtype=np.float64)[ordered]
        errors = translation[ordered]
        rotations = rotation[ordered]
        selected_local = int(np.argmax(selection))
        oracle_local = int(np.argmin(errors))
        ranks = _rank_descending(selection)
        selected_coverage = float(
            np.asarray(score["independent_evidence_coverages"], dtype=np.float64)[ordered][selected_local]
        )
        oracle_coverage = float(
            np.asarray(score["independent_evidence_coverages"], dtype=np.float64)[ordered][oracle_local]
        )
        selected_error = float(errors[selected_local])
        oracle_error = float(errors[oracle_local])
        categories: list[str] = []
        if selected_error >= float(tail_threshold_m):
            categories.append("catastrophic_selected_pose")
        if oracle_error > float(good_pose_threshold_m):
            categories.append("no_good_pose_in_frozen_hypotheses")
        if selected_coverage < float(low_coverage_threshold):
            categories.append("low_selected_evidence_coverage")
        if selected_error >= float(tail_threshold_m) and selected_coverage >= float(low_coverage_threshold):
            categories.append("self_consistent_wrong_pose_under_aggregate_evidence")
        if not mode_status["spatial_modes_available_for_this_score"]:
            categories.append("rgb_spatial_modes_not_present")
        if not categories:
            categories.append("non_tail")
        row: dict[str, Any] = {
            "query_id": query_id,
            "split_name": split_name,
            "evaluation_label": label,
            "hypothesis_count": int(len(ordered)),
            "selected_hypothesis_index": int(score["hypothesis_indices"][ordered][selected_local]),
            "oracle_hypothesis_index": int(score["hypothesis_indices"][ordered][oracle_local]),
            "selected_translation_error_m_TARGET_ONLY": selected_error,
            "selected_rotation_error_deg_TARGET_ONLY": float(rotations[selected_local]),
            "oracle_translation_error_m_TARGET_ONLY": oracle_error,
            "oracle_rotation_error_deg_TARGET_ONLY": float(rotations[oracle_local]),
            "oracle_score_rank_TARGET_ONLY": int(ranks[oracle_local]),
            "selected_score": float(selection[selected_local]),
            "oracle_score": float(selection[oracle_local]),
            "selected_vs_second_score_margin": _score_margin(selection, selected_local),
            "oracle_vs_selected_score_gap": float(selection[oracle_local] - selection[selected_local]),
            "selected_evidence_coverage": selected_coverage,
            "oracle_evidence_coverage": oracle_coverage,
            "selected_effective_point_count": int(
                np.asarray(score["independent_effective_point_counts"], dtype=np.int64)[ordered][selected_local]
            ),
            "oracle_effective_point_count": int(
                np.asarray(score["independent_effective_point_counts"], dtype=np.int64)[ordered][oracle_local]
            ),
            "verification_point_count": int(
                np.asarray(score["verification_point_counts"], dtype=np.int64)[ordered][selected_local]
            ),
            "tail_categories": categories,
        }
        for name, field in score_names.items():
            values = np.asarray(score[field], dtype=np.float64)[ordered]
            row[f"selected_{name}"] = float(values[selected_local])
            row[f"oracle_{name}"] = float(values[oracle_local])
            row[f"oracle_minus_selected_{name}"] = float(values[oracle_local] - values[selected_local])
        rows.append(row)

    tail_rows = [
        row for row in rows if row["selected_translation_error_m_TARGET_ONLY"] >= float(tail_threshold_m)
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["selected_translation_error_m_TARGET_ONLY"]),
            str(row["query_id"]),
        ),
    )[: int(top_n)]
    return {
        "stage": "audit_independent_pose_tail_target_only",
        "format": REPORT_FORMAT,
        "score_inputs_target_free": True,
        "target_join_after_score_generation": True,
        "score_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(Path(path))}
            for path in score_artifacts
        ],
        "target_artifact": {
            "path": str(target_artifact),
            "sha256": file_sha256_short(Path(target_artifact)),
        },
        "thresholds": {
            "tail_threshold_m": float(tail_threshold_m),
            "good_pose_threshold_m": float(good_pose_threshold_m),
            "low_coverage_threshold": float(low_coverage_threshold),
        },
        "mode_status": mode_status,
        "summary": {
            "query_count": int(len(rows)),
            "catastrophic_selected_count": int(len(tail_rows)),
            "good_oracle_available_count": int(
                sum(
                    row["oracle_translation_error_m_TARGET_ONLY"]
                    <= float(good_pose_threshold_m)
                    for row in rows
                )
            ),
            "catastrophic_with_good_oracle_count": int(
                sum(
                    row["selected_translation_error_m_TARGET_ONLY"] >= float(tail_threshold_m)
                    and row["oracle_translation_error_m_TARGET_ONLY"] <= float(good_pose_threshold_m)
                    for row in rows
                )
            ),
            "catastrophic_with_high_coverage_count": int(
                sum(
                    row["selected_translation_error_m_TARGET_ONLY"] >= float(tail_threshold_m)
                    and row["selected_evidence_coverage"] >= float(low_coverage_threshold)
                    for row in rows
                )
            ),
        },
        "tail_queries_TARGET_ONLY": tail_rows,
        "worst_queries_TARGET_ONLY": ranked,
        "all_queries_TARGET_ONLY": rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "split_name",
        "evaluation_label",
        "hypothesis_count",
        "selected_hypothesis_index",
        "oracle_hypothesis_index",
        "selected_translation_error_m_TARGET_ONLY",
        "oracle_translation_error_m_TARGET_ONLY",
        "oracle_score_rank_TARGET_ONLY",
        "selected_score",
        "oracle_score",
        "selected_vs_second_score_margin",
        "oracle_vs_selected_score_gap",
        "selected_evidence_coverage",
        "oracle_evidence_coverage",
        "selected_effective_point_count",
        "oracle_effective_point_count",
        "tail_categories",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["tail_categories"] = ";".join(row["tail_categories"])
            writer.writerow({name: serialized.get(name) for name in fields})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    if (output_json.exists() or output_csv.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite tail audit output")
    score_paths = tuple(
        Path(item.strip()) for item in str(args.score_artifacts).split(",") if item.strip()
    )
    report = audit_independent_pose_tail(
        score_artifacts=score_paths,
        target_artifact=Path(args.target_artifact),
        tail_threshold_m=float(args.tail_threshold_m),
        good_pose_threshold_m=float(args.good_pose_threshold_m),
        low_coverage_threshold=float(args.low_coverage_threshold),
        top_n=int(args.top_n),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_csv(output_csv, report["all_queries_TARGET_ONLY"])
    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
