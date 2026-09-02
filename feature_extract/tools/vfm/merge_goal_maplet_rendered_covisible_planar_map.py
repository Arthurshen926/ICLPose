"""Merge rendered finite-plane fragments from mapping-only covisibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
    PrimitiveSurfaceTable,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.rendered_plane_covisible_merge import (
    merge_rendered_covisible_planes,
)
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import (
    RenderedPlaneObservations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface_table", type=Path, required=True)
    parser.add_argument("--input_map", type=Path, required=True)
    parser.add_argument("--input_lineage", type=Path, required=True)
    parser.add_argument("--contributors", type=Path, required=True)
    parser.add_argument("--observation_dir", type=Path, required=True)
    parser.add_argument("--image_ids_file", type=Path)
    parser.add_argument("--output_map", type=Path, required=True)
    parser.add_argument("--output_lineage", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--minimum_covisible_views", type=int, default=2)
    parser.add_argument("--normal_degrees", type=float, default=6.0)
    parser.add_argument("--reciprocal_plane_distance_m", type=float, default=0.05)
    parser.add_argument("--maximum_refit_rms_m", type=float, default=0.05)
    parser.add_argument("--maximum_refit_p95_m", type=float, default=0.08)
    parser.add_argument("--fit_normal_degrees", type=float, default=15.0)
    parser.add_argument("--maximum_aggregate_rendered_rms_m", type=float, default=0.07)
    args = parser.parse_args()
    for output in (args.output_map, args.output_lineage, args.output_summary):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    table = PrimitiveSurfaceTable.load_npz(args.surface_table)
    planar = GeometryNativePlanarMap.load_npz(
        args.input_map, primitive_count=table.primitive_ids.size,
    )
    with np.load(args.input_lineage, allow_pickle=False) as data:
        offsets = np.asarray(data["plane_observation_offsets"], np.int64)
        rows = np.asarray(data["plane_observation_rows"], np.int64)
    selected_ids = None
    if args.image_ids_file is not None:
        selected_ids = {
            line.strip().replace("/", "__")
            for line in args.image_ids_file.read_text().splitlines() if line.strip()
        }
    paths = sorted(args.contributors.glob("*.npz"))
    if selected_ids is not None:
        paths = [path for path in paths if path.stem in selected_ids]
        if len(paths) != len(selected_ids):
            raise RuntimeError("selected contributor inventory is incomplete")
    observation_parts = {name: [] for name in (
        "pixel_counts", "point_sum_world", "point_second_moment_world", "residual_p95_m",
    )}
    observation_paths = sorted(args.observation_dir.glob("*.planes.npz"))
    if selected_ids is not None:
        observation_paths = [
            path for path in observation_paths
            if path.name[:-len(".planes.npz")] in selected_ids
        ]
        if len(observation_paths) != len(selected_ids):
            raise RuntimeError("selected plane-observation inventory is incomplete")
    for path in observation_paths:
        observation, _ = RenderedPlaneObservations.load_npz(
            path, primitive_count=table.primitive_ids.size,
        )
        for name in observation_parts:
            observation_parts[name].append(np.asarray(getattr(observation, name)))
    observation_stats = {
        name: np.concatenate(values, axis=0) for name, values in observation_parts.items()
    }
    merged, lineage, audit = merge_rendered_covisible_planes(
        table, planar, offsets, rows, paths,
        observation_stats=observation_stats,
        minimum_covisible_views=args.minimum_covisible_views,
        normal_degrees=args.normal_degrees,
        reciprocal_plane_distance_m=args.reciprocal_plane_distance_m,
        maximum_refit_rms_m=args.maximum_refit_rms_m,
        maximum_refit_p95_m=args.maximum_refit_p95_m,
        fit_normal_degrees=args.fit_normal_degrees,
        maximum_aggregate_rendered_rms_m=args.maximum_aggregate_rendered_rms_m,
    )
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    merged.save_npz(args.output_map)
    np.savez_compressed(args.output_lineage, **lineage)
    payload = {
        "artifact_type": "goal_maplet_rendered_covisible_plane_merge_report_v1",
        **audit,
        "input_map_file_sha256": file_sha256(args.input_map),
        "input_lineage_file_sha256": file_sha256(args.input_lineage),
        "surface_table_file_sha256": file_sha256(args.surface_table),
        "output_map_file_sha256": file_sha256(args.output_map),
        "output_lineage_file_sha256": file_sha256(args.output_lineage),
        "uses_query_or_ground_truth": False,
        "uses_mapping_rgb_or_pose": False,
        "component_complete_linkage_coplanarity": True,
        "observation_manifest_file_sha256": file_sha256(args.observation_dir / "manifest.json"),
        "image_ids_file_sha256": (
            None if args.image_ids_file is None else file_sha256(args.image_ids_file)
        ),
        "configuration": {
            "minimum_covisible_views": args.minimum_covisible_views,
            "normal_degrees": args.normal_degrees,
            "reciprocal_plane_distance_m": args.reciprocal_plane_distance_m,
            "maximum_refit_rms_m": args.maximum_refit_rms_m,
            "maximum_refit_p95_m": args.maximum_refit_p95_m,
            "fit_normal_degrees": args.fit_normal_degrees,
            "maximum_aggregate_rendered_rms_m": args.maximum_aggregate_rendered_rms_m,
        },
    }
    args.output_summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
