"""Join frozen pose-only inference results to StMarysChurch GT exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_jsonl", required=True)
    parser.add_argument("--ground_truth_pose_file", required=True)
    parser.add_argument("--expected_query_list", required=True)
    parser.add_argument("--baseline_results_jsonl", default="")
    parser.add_argument(
        "--mapping_pose_file",
        default="",
        help="Optional train-pose file for a paired VFM top-support retrieval-pose baseline.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--errors_jsonl", required=True)
    return parser.parse_args(argv)


def _read_results(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = dict(json.loads(line))
        image_id = str(record["image_id"])
        if image_id in records:
            raise ValueError(f"duplicate result image ID: {image_id}")
        records[image_id] = record
    return records


def _query_ids(path: Path) -> list[str]:
    output = [
        line.strip().split()[0]
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(output) != len(set(output)):
        raise ValueError("expected query list contains duplicates")
    return output


def _pose_error(estimated_w2c: np.ndarray, target_w2c: np.ndarray) -> tuple[float, float]:
    estimated = np.asarray(estimated_w2c, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_w2c, dtype=np.float64).reshape(4, 4)
    estimated_center = -estimated[:3, :3].T @ estimated[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation_m = float(np.linalg.norm(estimated_center - target_center))
    relative = estimated[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return translation_m, float(np.degrees(np.arccos(cosine)))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _evaluate(
    results: dict[str, dict[str, object]],
    query_ids: list[str],
    target_by_image: dict[str, np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    missing = sorted(set(query_ids) - set(results))
    extra = sorted(set(results) - set(query_ids))
    if missing or extra:
        raise ValueError(
            f"result/query ID mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    translations: list[float] = []
    rotations: list[float] = []
    errors: list[dict[str, object]] = []
    thresholds = (
        ("5cm_5deg", 0.05, 5.0),
        ("10cm_5deg", 0.10, 5.0),
        ("25cm_10deg", 0.25, 10.0),
        ("50cm_10deg", 0.50, 10.0),
        ("5m_10deg", 5.0, 10.0),
    )
    threshold_counts = {name: 0 for name, _distance, _angle in thresholds}
    success_count = 0
    for image_id in query_ids:
        result = results[image_id]
        success = bool(result.get("success", False)) and result.get("pose_w2c") is not None
        translation = None
        rotation = None
        if success:
            success_count += 1
            translation, rotation = _pose_error(
                np.asarray(result["pose_w2c"], dtype=np.float64),
                target_by_image[image_id],
            )
            translations.append(float(translation))
            rotations.append(float(rotation))
            for name, distance, angle in thresholds:
                threshold_counts[name] += int(
                    float(translation) <= float(distance) and float(rotation) <= float(angle)
                )
        errors.append(
            {
                "image_id": image_id,
                "success": bool(success),
                "translation_m": translation,
                "rotation_deg": rotation,
                "failure_reason": result.get("failure_reason"),
            }
        )
    count = len(query_ids)
    return (
        {
            "query_count": int(count),
            "success_count": int(success_count),
            "success_rate": float(success_count / max(count, 1)),
            "translation_m_successful": _stats(translations),
            "rotation_deg_successful": _stats(rotations),
            "threshold_recall_all_queries": {
                name: float(value / max(count, 1)) for name, value in threshold_counts.items()
            },
        },
        errors,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    query_ids = _query_ids(Path(args.expected_query_list))
    target_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.ground_truth_pose_file))
    }
    if set(query_ids) != set(target_by_image):
        raise ValueError("frozen query list and GT pose IDs differ")
    results = _read_results(Path(args.results_jsonl))
    metrics, errors = _evaluate(results, query_ids, target_by_image)
    output: dict[str, object] = {
        "stage": "evaluate_2dgs_surface_localization",
        "new_method": metrics,
        "paired_baseline": None,
        "retrieval_pose_baseline": None,
        "frozen_inputs": {
            "results_jsonl": str(args.results_jsonl),
            "results_sha256": file_sha256_short(Path(args.results_jsonl)),
            "ground_truth_pose_file": str(args.ground_truth_pose_file),
            "ground_truth_pose_file_sha256": file_sha256_short(
                Path(args.ground_truth_pose_file)
            ),
            "expected_query_list": str(args.expected_query_list),
            "expected_query_list_sha256": file_sha256_short(
                Path(args.expected_query_list)
            ),
        },
    }
    if str(args.mapping_pose_file):
        mapping_pose_by_image = {
            record.image_id: record.pose_w2c
            for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
        }
        retrieval_results: dict[str, dict[str, object]] = {}
        for image_id, result in results.items():
            support_views = list(
                dict(result.get("diagnostics", {})).get("vfm_support_views", [])
            )
            support_view = str(support_views[0]) if support_views else ""
            support_pose = mapping_pose_by_image.get(support_view)
            retrieval_results[image_id] = {
                "image_id": image_id,
                "success": support_pose is not None,
                "pose_w2c": (
                    support_pose.reshape(-1).tolist()
                    if support_pose is not None
                    else None
                ),
                "failure_reason": (
                    None if support_pose is not None else "missing_vfm_support_pose"
                ),
            }
        retrieval_metrics, retrieval_errors = _evaluate(
            retrieval_results,
            query_ids,
            target_by_image,
        )
        new_error_by_id = {str(row["image_id"]): row for row in errors}
        retrieval_error_by_id = {
            str(row["image_id"]): row for row in retrieval_errors
        }
        both_success = [
            image_id
            for image_id in query_ids
            if bool(new_error_by_id[image_id]["success"])
            and bool(retrieval_error_by_id[image_id]["success"])
        ]
        output["retrieval_pose_baseline"] = {
            "metrics": retrieval_metrics,
            "both_success_count": int(len(both_success)),
            "new_minus_retrieval_translation_m": _stats(
                [
                    float(new_error_by_id[image_id]["translation_m"])
                    - float(retrieval_error_by_id[image_id]["translation_m"])
                    for image_id in both_success
                ]
            ),
            "new_minus_retrieval_rotation_deg": _stats(
                [
                    float(new_error_by_id[image_id]["rotation_deg"])
                    - float(retrieval_error_by_id[image_id]["rotation_deg"])
                    for image_id in both_success
                ]
            ),
            "mapping_pose_file": str(args.mapping_pose_file),
            "mapping_pose_file_sha256": file_sha256_short(
                Path(args.mapping_pose_file)
            ),
        }
    if str(args.baseline_results_jsonl):
        baseline_results = _read_results(Path(args.baseline_results_jsonl))
        baseline_metrics, baseline_errors = _evaluate(
            baseline_results,
            query_ids,
            target_by_image,
        )
        new_error_by_id = {str(row["image_id"]): row for row in errors}
        baseline_error_by_id = {str(row["image_id"]): row for row in baseline_errors}
        both_success = [
            image_id
            for image_id in query_ids
            if bool(new_error_by_id[image_id]["success"])
            and bool(baseline_error_by_id[image_id]["success"])
        ]
        translation_delta = [
            float(new_error_by_id[image_id]["translation_m"])
            - float(baseline_error_by_id[image_id]["translation_m"])
            for image_id in both_success
        ]
        rotation_delta = [
            float(new_error_by_id[image_id]["rotation_deg"])
            - float(baseline_error_by_id[image_id]["rotation_deg"])
            for image_id in both_success
        ]
        output["paired_baseline"] = {
            "metrics": baseline_metrics,
            "both_success_count": int(len(both_success)),
            "new_minus_baseline_translation_m": _stats(translation_delta),
            "new_minus_baseline_rotation_deg": _stats(rotation_delta),
            "baseline_results_jsonl": str(args.baseline_results_jsonl),
            "baseline_results_sha256": file_sha256_short(
                Path(args.baseline_results_jsonl)
            ),
        }
    output_path = Path(args.output_json)
    errors_path = Path(args.errors_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with errors_path.open("w") as handle:
        for row in errors:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
