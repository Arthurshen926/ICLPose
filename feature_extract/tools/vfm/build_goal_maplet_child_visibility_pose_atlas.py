"""Build a feature-free child/coarse-layout visibility pose atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    build_child_visibility_pose_atlas,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output_atlas", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_view_count", type=int, default=1487)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--maximum_global_children", type=int, default=1024)
    parser.add_argument("--maximum_children_per_cell", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_atlas)
    report_path = Path(args.output_json)
    if (output.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite visibility-atlas outputs")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    contributors = sorted(Path(args.contributors).glob("*.npz"))
    if len(contributors) != int(args.expected_view_count):
        raise ValueError(
            f"expected {args.expected_view_count} contributors, found {len(contributors)}"
        )
    atlas = build_child_visibility_pose_atlas(
        physical,
        contributors,
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
        maximum_global_children=int(args.maximum_global_children),
        maximum_children_per_cell=int(args.maximum_children_per_cell),
        metadata={
            "source_contributor_directory": str(Path(args.contributors).resolve()),
            "physical_map_path": str(Path(args.physical_map).resolve()),
            "maximum_global_children": int(args.maximum_global_children),
            "maximum_children_per_cell": int(args.maximum_children_per_cell),
        },
    )
    atlas.save_npz(output)
    report = {
        "artifact_type": "goal_maplet_child_visibility_pose_atlas_build_v3",
        "content_hash_includes_schema_score_grid_and_physical_map": True,
        "output_atlas": str(output.resolve()),
        "content_sha256": atlas.content_sha256,
        "physical_map_sha256": atlas.physical_map_sha256,
        "view_count": atlas.view_count,
        "child_count": atlas.child_count,
        "grid_rows": atlas.grid_rows,
        "grid_cols": atlas.grid_cols,
        "layout_normalization": "joint_cell_child_probability_after_per_cell_truncation",
        "global_relation_count": int(atlas.global_child_rows.size),
        "layout_relation_count": int(atlas.layout_keys.size),
        "mean_global_children_per_view": float(
            atlas.global_child_rows.size / max(atlas.view_count, 1)
        ),
        "mean_layout_children_per_view": float(
            atlas.layout_keys.size / max(atlas.view_count, 1)
        ),
        "method_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_descriptors": False,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "uses_alike": False,
            "uses_pnp": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "output_is_final_pose": False,
            "output_role": "multi_modal_visibility_chart_centres",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
