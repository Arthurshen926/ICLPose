"""Freeze unused held cameras for the next physically isolated reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.chart_complement_held_plan import (
    build_complement_held_preexecution_plan,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_only_plan", type=Path, required=True)
    parser.add_argument("--expected_pose_only_plan_content_sha256", required=True)
    parser.add_argument("--intended_source_chart_plan_content_sha256", required=True)
    parser.add_argument("--planned_isolation_root", type=Path, required=True)
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--held_route", default="seq1")
    parser.add_argument("--target_count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite complement-held plan")
    plan = build_complement_held_preexecution_plan(
        args.pose_only_plan,
        expected_pose_only_plan_content_sha256=(
            args.expected_pose_only_plan_content_sha256
        ),
        intended_source_chart_plan_content_sha256=(
            args.intended_source_chart_plan_content_sha256
        ),
        planned_isolation_root=args.planned_isolation_root,
        matcha_repo=args.matcha_repo,
        route=args.held_route,
        target_count=args.target_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "artifact_type": plan["artifact_type"],
                "output": str(args.output.resolve()),
                "output_file_sha256": file_sha256(args.output),
                "output_content_sha256": plan["content_sha256"],
                "fresh_held_indices": plan["fresh_held_indices"],
                "fresh_held_ordered_names": plan["fresh_held_ordered_names"],
                "fresh_held_disjoint_from_superseded_diagnostic": plan[
                    "fresh_held_disjoint_from_superseded_diagnostic"
                ],
                "selection_frozen_before_new_held_geometry": plan[
                    "selection_frozen_before_new_held_geometry"
                ],
                "physical_isolation_input_builder_command": plan[
                    "physical_isolation_input_builder_command"
                ],
                "sfm_preexecution_contract_builder_command": plan[
                    "sfm_preexecution_contract_builder_command"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
