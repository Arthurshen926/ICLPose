"""Build a typed image-free graph for Goal-Maplet context parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.typed_graph import build_typed_parent_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--output_graph", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--exclude_trajectories", nargs="*", default=["seq11", "seq3", "seq5", "seq13"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_graph), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite typed graph")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    excluded = set(str(value) for value in args.exclude_trajectories)
    paths = []
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            item = json.loads(str(data["metadata_json"].item()))
        if str(item.get("trajectory_id", "")) not in excluded:
            paths.append(path)
    graph = build_typed_parent_graph(physical, field, paths)
    graph.save_npz(output)
    type_count = {
        str(kind): int((graph.edge_type == kind).sum()) for kind in sorted(set(graph.edge_type.tolist()))
    }
    report = {
        "stage": "build_goal_maplet_typed_graph",
        "output_graph": str(output),
        "typed_graph_sha256": graph.content_sha256,
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "parent_count": int(graph.parent_view_count.size),
        "edge_count": int(graph.edge_source.size),
        "edge_type_count": type_count,
        "observed_parent_fraction": float((graph.parent_view_count > 0).mean()),
        "metadata": dict(graph.metadata),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
