"""Build pose-bearing mapping-view nodes without retaining mapping images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.mapping_view_graph import build_mapping_view_graph
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--output_graph", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--exclude_trajectories", nargs="+", default=[])
    parser.add_argument("--maximum_parents_per_view", type=int, default=128)
    parser.add_argument("--minimum_parent_mass_fraction", type=float, default=5.0e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_graph)
    report_path = Path(args.output_json)
    if (output.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite mapping-view graph artifact")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    excluded = set(str(value) for value in args.exclude_trajectories)
    contributors = []
    for path in sorted(Path(args.contributors).glob("*.npz")):
        trajectory = path.name.split("__", 1)[0]
        if trajectory not in excluded:
            contributors.append(path)
    graph = build_mapping_view_graph(
        physical,
        field.content_sha256,
        contributors,
        maximum_parents_per_view=int(args.maximum_parents_per_view),
        minimum_parent_mass_fraction=float(args.minimum_parent_mass_fraction),
        metadata={
            "excluded_trajectory_ids": sorted(excluded),
            "source_contributor_count": len(contributors),
        },
    )
    graph.save_npz(output)
    report = {
        "stage": "build_goal_maplet_mapping_view_graph",
        "artifact_type": "goal_maplet_mapping_view_graph_v1",
        "output_graph": str(output.resolve()),
        "content_sha256": graph.content_sha256,
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "view_node_count": int(graph.poses_w2c.shape[0]),
        "relation_count": int(graph.parent_rows.size),
        "mean_parent_relations_per_view": float(
            graph.parent_rows.size / max(graph.poses_w2c.shape[0], 1)
        ),
        "excluded_trajectory_ids": sorted(excluded),
        "map_contract": {
            "stored_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_tracks": False,
            "uses_point_correspondences": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
