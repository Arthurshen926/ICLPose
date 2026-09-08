"""Aggregate hash-bound post-label pose failure audits without changing labels.

The input audits are immutable diagnostics produced after all pose candidates
were frozen.  This utility verifies every canonical content hash, rejects
duplicate query names, and emits the single 438-query failure ledger used by
subsequent method design.  It is not a selector or a training artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


EXPECTED_ARTIFACT = "goal_maplet_pose_failure_stage_postlabel_audit_v1"


def _load_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text())
    claimed = report.pop("content_sha256", None)
    actual = canonical_json_sha256(report)
    report["content_sha256"] = claimed
    if report.get("artifact_type") != EXPECTED_ARTIFACT:
        raise ValueError(f"unexpected input artifact type: {path}")
    if claimed != actual:
        raise ValueError(f"input canonical content hash differs: {path}")
    if report.get("selection_or_training_eligible") is not False:
        raise ValueError(f"input must remain diagnostic-only: {path}")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != int(report.get("query_count", -1)):
        raise ValueError(f"input row count differs: {path}")
    return report


def aggregate(paths: list[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one input report is required")
    reports = [_load_report(path) for path in paths]
    thresholds = [report["coarse_threshold"] for report in reports]
    if any(value != thresholds[0] for value in thresholds[1:]):
        raise ValueError("coarse thresholds differ across input reports")
    oracle_definitions = [report["oracle_definition"] for report in reports]
    if any(value != oracle_definitions[0] for value in oracle_definitions[1:]):
        raise ValueError("oracle definitions differ across input reports")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for report in reports:
        split = str(report["split_name"])
        for source_row in report["rows"]:
            row = dict(source_row)
            name = str(row["name"])
            key = f"{split}:{name}"
            if key in seen:
                raise ValueError(f"duplicate split/query row: {key}")
            seen.add(key)
            row["split_name"] = split
            rows.append(row)

    all_counts = Counter(str(row["final_failure_category"]) for row in rows)
    failures = [row for row in rows if not bool(row["selected_is_2m_45deg_hit"])]
    failure_counts = Counter(str(row["final_failure_category"]) for row in failures)
    init_failures = [
        row for row in failures
        if row["final_failure_category"] == "hard_coordinate_or_pnp_initialization_failure"
    ]

    def distribution(field: str) -> dict[str, float | int | None]:
        values = np.asarray([row[field] for row in init_failures], np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"count": 0, "minimum": None, "median": None, "maximum": None}
        return {
            "count": int(len(values)),
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }

    summary: dict[str, object] = {
        "artifact_type": "goal_maplet_pose_failure_stage_aggregate_postlabel_audit_v1",
        "evaluation_role": "POSTLABEL_DIAGNOSTIC_ONLY_NOT_SELECTION_OR_TRAINING",
        "coarse_threshold": thresholds[0],
        "oracle_definition": oracle_definitions[0],
        "query_count": int(len(rows)),
        "selected_coarse_success_count": int(len(rows) - len(failures)),
        "selected_coarse_failure_count": int(len(failures)),
        "all_category_counts": dict(sorted(all_counts.items())),
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "initialization_failure_existing_support": {
            "query_count": int(len(init_failures)),
            "oracle_candidate_row_count": distribution("oracle_candidate_row_count"),
            "oracle_physical_plane_count": distribution("oracle_physical_plane_count"),
            "minimum_candidate_gt_reprojection_px": distribution(
                "minimum_candidate_gt_reprojection_px"
            ),
        },
        "input_lineage": [
            {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "content_sha256": report["content_sha256"],
                "split_name": report["split_name"],
                "query_count": report["query_count"],
            }
            for path, report in zip(paths, reports)
        ],
        "correct_chart_vs_correct_uv_separable_without_depth_gt": False,
        "candidate_absence_category_combines_chart_retrieval_and_within_chart_uv_support": True,
        "selection_or_training_eligible": False,
        "query_pose_or_ground_truth_read": True,
        "rows": rows,
    }
    summary["content_sha256"] = canonical_json_sha256(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite aggregate failure audit")
    report = aggregate(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
