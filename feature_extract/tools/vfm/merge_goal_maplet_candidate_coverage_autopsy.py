"""Merge disjoint held-fold Goal-Maplet candidate-coverage autopsies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.audit_goal_maplet_candidate_coverage import (
    _classify,
    _summary,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _merge(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("no candidate-coverage autopsy reports")
    if any(
        report.get("stage") != "goal_maplet_candidate_coverage_autopsy_g20_3"
        for report in reports
    ):
        raise ValueError("input is not a G20.3 candidate-coverage autopsy")
    physical = {str(report.get("physical_map_sha256", "")) for report in reports}
    fields = [str(report.get("canonical_field_sha256", "")) for report in reports]
    if len(physical) != 1 or "" in physical:
        raise ValueError("autopsy folds use different physical maps")
    if len(fields) != len(set(fields)) or any(not value for value in fields):
        raise ValueError("autopsy folds reused or omitted a canonical field")
    held = [
        str(value)
        for report in reports
        for value in report.get("heldout_trajectories", ())
    ]
    if len(held) != len(set(held)):
        raise ValueError("autopsy held trajectories overlap")
    rows = [dict(row) for report in reports for row in report.get("rows", ())]
    for row in rows:
        row["classification"] = _classify(row)
    image_ids = [str(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("autopsy query rows overlap")
    return {
        "query_count": len(rows),
        "heldout_trajectories": sorted(held),
        "physical_map_sha256": next(iter(physical)),
        "canonical_field_sha256_per_fold": fields,
        "feature_pipeline_outer_crossfit": True,
        "geometry_was_outer_crossfit": False,
        "fixed_candidate_budget": 32,
        "phase_used_for_pose_generation": False,
        "summary": _summary(rows),
        "per_trajectory": {
            trajectory: _summary([
                row for row in rows if str(row["trajectory_id"]) == trajectory
            ])
            for trajectory in sorted(set(held))
        },
        "baseline_no_one_m_queries": [
            str(row["image_id"])
            for row in rows
            if not bool(
                row["candidate_ladder"]["baseline_post_nms"]["one_m_available"]
            )
        ],
        "baseline_no_strict_queries": [
            str(row["image_id"])
            for row in rows
            if not bool(
                row["candidate_ladder"]["baseline_post_nms"]["strict_available"]
            )
        ],
        "rows": sorted(rows, key=lambda row: str(row["image_id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged candidate autopsy")
    paths = [Path(value) for value in args.inputs]
    reports = [json.loads(path.read_text()) for path in paths]
    result = {
        "stage": "goal_maplet_candidate_coverage_autopsy_g20_3_merged",
        "fold_inputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        **_merge(reports),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": result["query_count"],
        "summary": result["summary"],
        "per_trajectory": result["per_trajectory"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
