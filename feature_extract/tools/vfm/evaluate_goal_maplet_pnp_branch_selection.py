"""Post-label evaluation of a frozen Top5/Top10 PnP branch-selection report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


ALLOWED_SELECTIONS = {
    "goal_maplet_pnp_pose_conditioned_view_context_selection_v1",
    "goal_maplet_pnp_pose_conditioned_global_radio_selection_v1",
    "goal_maplet_pnp_pose_conditioned_spatial_radio_selection_v1",
}


def _good(row: dict[str, object], translation: float, rotation: float) -> bool:
    return (
        bool(row.get("usable"))
        and float(row["translation_error_m"]) <= translation
        and float(row["rotation_error_deg"]) <= rotation
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--top5_evaluation", type=Path, required=True)
    parser.add_argument("--top10_evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite branch selection evaluation")
    selection = json.loads(args.selection.read_text())
    top5 = json.loads(args.top5_evaluation.read_text())
    top10 = json.loads(args.top10_evaluation.read_text())
    if (
        selection.get("artifact_type") not in ALLOWED_SELECTIONS
        or selection.get("query_pose_or_ground_truth_read") is not False
    ):
        raise ValueError("branch selection is not a pose-free supported artifact")
    rows5 = {str(row["name"]): row for row in top5["rows"]}
    rows10 = {str(row["name"]): row for row in top10["rows"]}
    rows = []
    for selected in selection["rows"]:
        name = str(selected["name"])
        if name not in rows5 or name not in rows10:
            raise ValueError("branch evaluations do not cover selection names")
        branch = int(selected["selected_branch"])
        source = rows10[name] if branch == 10 else rows5[name]
        rows.append({
            "name": name,
            "selected_branch": branch,
            "usable": bool(source["usable"]),
            "translation_error_m": source.get("translation_error_m"),
            "rotation_error_deg": source.get("rotation_error_deg"),
        })
    if len(rows) != int(selection.get("query_count", -1)):
        raise ValueError("selection query count differs")
    report = {
        "artifact_type": "goal_maplet_pnp_pose_free_branch_selection_postlabel_evaluation_v1",
        "selection_artifact_type": selection["artifact_type"],
        "selection_file_sha256": file_sha256(args.selection),
        "selection_content_sha256": selection.get("content_sha256"),
        "top5_evaluation_file_sha256": file_sha256(args.top5_evaluation),
        "top10_evaluation_file_sha256": file_sha256(args.top10_evaluation),
        "selection_frozen_before_pose_labels_opened": True,
        "query_count": int(len(rows)),
        "selected_top10_count": int(sum(row["selected_branch"] == 10 for row in rows)),
        "recall_2m45": float(np.mean([_good(row, 2.0, 45.0) for row in rows])),
        "recall_1m10": float(np.mean([_good(row, 1.0, 10.0) for row in rows])),
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
