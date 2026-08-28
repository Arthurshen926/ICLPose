"""Build a finite planar map directly from raw 2DGS primitive geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    PrimitiveSurfaceTable, extract_geometry_native_planar_map,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical_map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum_hypotheses", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--adjacency_normal_degrees", type=float, default=12.0)
    parser.add_argument("--adjacency_plane_distance_m", type=float, default=.05)
    parser.add_argument("--adjacency_support_gap_m", type=float, default=.10)
    parser.add_argument("--fit_normal_degrees", type=float, default=15.0)
    parser.add_argument("--fit_plane_distance_m", type=float, default=.06)
    parser.add_argument("--minimum_members", type=int, default=6)
    parser.add_argument("--minimum_support_area_m2", type=float, default=.10)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("planar map outputs must be new")
    started = time.perf_counter()
    physical = GoalMapletPhysicalMap.load_npz(args.physical_map)
    table = PrimitiveSurfaceTable.from_physical_map(physical)
    planar = extract_geometry_native_planar_map(
        table, maximum_hypotheses=int(args.maximum_hypotheses),
        neighbors=int(args.neighbors),
        adjacency_normal_degrees=float(args.adjacency_normal_degrees),
        adjacency_plane_distance_m=float(args.adjacency_plane_distance_m),
        adjacency_support_gap_m=float(args.adjacency_support_gap_m),
        fit_normal_degrees=float(args.fit_normal_degrees),
        fit_plane_distance_m=float(args.fit_plane_distance_m),
        minimum_members=int(args.minimum_members),
        minimum_support_area_m2=float(args.minimum_support_area_m2),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    planar.save_npz(args.output)
    loaded = type(planar).load_npz(args.output, primitive_count=table.primitive_ids.size)
    order = loaded.support_area_m2.argsort()[::-1]
    summary = {
        "artifact_type": "goal_maplet_geometry_native_planar_map_build_v1",
        "physical_map": str(args.physical_map.resolve()),
        "physical_map_file_sha256": file_sha256(args.physical_map),
        "planar_map": str(args.output.resolve()),
        "planar_map_file_sha256": file_sha256(args.output),
        "planar_map_arrays_sha256": loaded.metadata["arrays_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
        "primitive_count": int(table.primitive_ids.size),
        "plane_count": int(loaded.plane_ids.size),
        "assigned_primitive_fraction": float(loaded.member_primitive_rows.size / table.primitive_ids.size),
        "support_area_m2": {
            "sum": float(loaded.support_area_m2.sum()),
            "median": float(__import__('numpy').median(loaded.support_area_m2)),
            "p90": float(__import__('numpy').quantile(loaded.support_area_m2, .9)),
            "maximum": float(loaded.support_area_m2.max(initial=0.0)),
        },
        "member_count": {
            "median": float(__import__('numpy').median(loaded.member_counts)),
            "p90": float(__import__('numpy').quantile(loaded.member_counts, .9)),
            "maximum": int(loaded.member_counts.max(initial=0)),
        },
        "largest_plane_rows": order[:20].astype(int).tolist(),
        "metadata": loaded.metadata,
    }
    summary["content_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
