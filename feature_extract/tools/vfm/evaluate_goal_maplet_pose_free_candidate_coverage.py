"""Evaluate frozen pose-free candidate coverage before any pose scoring.

The input pool was generated without query pose or ground truth.  Its direct
candidate dataset then opened the target pose only after freezing the complete
ordered candidate payload.  This evaluator consumes only those post-freeze
error labels and therefore measures an acquisition/region upper bound, never
searched recall or localization success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_pose_free_candidate_coverage_v1"
JOINT_THRESHOLDS = {
    "strict_0_5m_5deg": (0.5, 5.0),
    "loose_1m_10deg": (1.0, 10.0),
    "region_2m_45deg": (2.0, 45.0),
}


def _coverage_rows(
    translation: np.ndarray,
    rotation: np.ndarray,
    valid: np.ndarray,
    budgets: tuple[int, ...],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    query_count = int(valid.shape[0])
    available = int(valid.shape[1])
    for requested in budgets:
        budget = min(int(requested), available)
        prefix_valid = valid[:, :budget]
        row: dict[str, object] = {
            "requested_candidate_budget": int(requested),
            "evaluated_candidate_budget": budget,
        }
        for name, (translation_limit, rotation_limit) in JOINT_THRESHOLDS.items():
            hit = np.any(
                prefix_valid
                & (translation[:, :budget] <= translation_limit)
                & (rotation[:, :budget] <= rotation_limit),
                axis=1,
            )
            row[name] = {
                "hits": int(np.sum(hit)),
                "query_count": query_count,
                "rate": float(np.mean(hit)),
            }
        for translation_limit in (0.5, 1.0, 2.0, 3.0, 5.0):
            hit = np.any(
                prefix_valid & (translation[:, :budget] <= translation_limit), axis=1,
            )
            row[f"translation_only_le_{translation_limit:g}m"] = {
                "hits": int(np.sum(hit)),
                "query_count": query_count,
                "rate": float(np.mean(hit)),
            }
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--candidate_budgets", default="1,4,8,16,32,64")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite candidate coverage report")
    budgets = tuple(sorted({int(value) for value in args.candidate_budgets.split(",")}))
    if not budgets or budgets[0] <= 0:
        raise ValueError("candidate budgets must be positive")

    dataset_rows: list[dict[str, object]] = []
    pooled_translation: list[np.ndarray] = []
    pooled_rotation: list[np.ndarray] = []
    pooled_valid: list[np.ndarray] = []
    for value in args.dataset:
        path = Path(value).resolve()
        arrays, metadata = load_pose_candidate_dataset(
            path, require_rendered_targets=False,
        )
        if metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA:
            raise ValueError("coverage input is not a direct frozen candidate dataset")
        if (
            metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True
            or metadata.get("candidate_pool_scores_consumed") is not False
            or metadata.get("pose_errors_computed_only_after_candidate_freeze") is not True
            or metadata.get(
                "nonanchor_candidates_preserve_pose_free_pool_exact_order"
            ) is not True
            or metadata.get(
                "gt_anchor_does_not_change_nonanchor_candidate_membership"
            ) is not True
            or metadata.get(
                "pose_free_pool_internal_duplicates_rejected_before_gt_join"
            ) is not True
        ):
            raise ValueError("candidate coverage labels were not opened after pool freeze")
        # Candidate zero is the diagnostic GT anchor.  It proves the renderer's
        # coordinate convention but would make acquisition coverage trivially
        # one, so every reported metric excludes it.
        translation = np.asarray(arrays["translation_m"], dtype=np.float64)[:, 1:]
        rotation = np.asarray(arrays["rotation_deg"], dtype=np.float64)[:, 1:]
        valid = np.asarray(arrays["candidate_valid"], dtype=bool)[:, 1:]
        if not np.all(np.sum(valid, axis=1) > 0):
            raise ValueError("candidate coverage input has an empty non-anchor row")
        joint = np.where(valid, np.maximum(translation / 2.0, rotation / 45.0), np.inf)
        best = np.argmin(joint, axis=1)
        row_index = np.arange(valid.shape[0])
        best_translation = translation[row_index, best]
        best_rotation = rotation[row_index, best]
        dataset_rows.append({
            "path": str(path),
            "file_sha256": file_sha256(path),
            "dataset_content_sha256": str(metadata["content_sha256"]),
            "query_count": int(valid.shape[0]),
            "query_routes": sorted({
                str(image_id).split("/", 1)[0]
                for image_id in np.asarray(arrays["image_ids"]).tolist()
            }),
            "candidate_semantics": str(metadata.get("candidate_semantics", "unknown")),
            "candidate_prefix_stable_across_budgets": bool(
                metadata.get("candidate_prefix_stable_across_budgets", False)
            ),
            "coverage_by_budget": _coverage_rows(
                translation, rotation, valid, budgets,
            ),
            "best_region_normalized_candidate": {
                "median_translation_m": float(np.median(best_translation)),
                "median_rotation_deg": float(np.median(best_rotation)),
                "p90_translation_m": float(np.percentile(best_translation, 90)),
                "p90_rotation_deg": float(np.percentile(best_rotation, 90)),
            },
        })
        pooled_translation.append(translation)
        pooled_rotation.append(rotation)
        pooled_valid.append(valid)

    maximum_width = max(value.shape[1] for value in pooled_valid)
    if any(value.shape[1] != maximum_width for value in pooled_valid):
        raise ValueError("pooled candidate datasets have different maximum budgets")
    translation = np.concatenate(pooled_translation, axis=0)
    rotation = np.concatenate(pooled_rotation, axis=0)
    valid = np.concatenate(pooled_valid, axis=0)
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "dataset_count": len(dataset_rows),
        "query_count": int(valid.shape[0]),
        "candidate_budgets": list(budgets),
        "candidate_zero_diagnostic_gt_anchor_excluded": True,
        "candidate_scores_consumed": False,
        "ground_truth_opened_only_after_pose_free_candidate_pool_freeze": True,
        "metric_semantics": (
            "pose_free_candidate_acquisition_upper_bound_not_searched_recall_"
            "not_localization_success_v1"
        ),
        "datasets": dataset_rows,
        "pooled_coverage_by_budget": _coverage_rows(
            translation, rotation, valid, budgets,
        ),
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "datasets"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
