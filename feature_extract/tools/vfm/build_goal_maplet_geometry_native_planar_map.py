"""Build a finite planar map directly from raw 2DGS primitive geometry."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    PrimitiveSurfaceTable, extract_geometry_native_planar_map,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.vfm_2dgs_mapping import _surface_tangent_axes_and_scales


def _primitive_table_from_2dgs_ply(
    path: Path, *, minimum_opacity: float, maximum_scale: float | None,
) -> PrimitiveSurfaceTable:
    source = load_gaussian_vfm_source_from_ply(path)
    if source.scale_xyz is None or source.rotation is None or source.normal is None:
        raise ValueError("direct planar-map construction requires oriented 2DGS disks")
    keep = np.asarray(source.opacity, np.float32) >= float(minimum_opacity)
    if maximum_scale is not None:
        keep &= np.asarray(source.scale, np.float32) <= float(maximum_scale)
    rows = np.flatnonzero(keep)
    if rows.size == 0:
        raise ValueError("2DGS filtering removed every primitive")
    normals = np.asarray(source.normal, np.float32)[rows]
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(
        source, rows, normals,
    )
    return PrimitiveSurfaceTable(
        primitive_ids=np.asarray(source.gaussian_indices, np.int64)[rows],
        centers=np.asarray(source.xyz, np.float64)[rows],
        tangent1=np.asarray(tangent1, np.float64),
        tangent2=np.asarray(tangent2, np.float64),
        normals=np.asarray(normals, np.float64),
        scale1=np.asarray(scale1, np.float64),
        scale2=np.asarray(scale2, np.float64),
        opacity=np.asarray(source.opacity, np.float64)[rows],
        metadata={
            "representation": "raw_oriented_2dgs_disks_without_parent_child_hierarchy",
            "source_2dgs_ply": str(path.resolve()),
            "source_2dgs_ply_file_sha256": file_sha256(path),
            "source_gaussian_count": int(source.xyz.shape[0]),
            "minimum_opacity": float(minimum_opacity),
            "maximum_scale": None if maximum_scale is None else float(maximum_scale),
            "uses_parent_child_partition": False,
            "uses_query_pose_or_ground_truth": False,
        },
    ).validated()


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--physical_map", type=Path)
    source.add_argument("--two_dgs_ply", type=Path)
    source.add_argument("--surface_table", type=Path)
    parser.add_argument("--surface_table_output", type=Path)
    parser.add_argument("--surface_min_opacity", type=float, default=.05)
    parser.add_argument("--surface_max_scale", type=float)
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
    if args.output.exists() or args.summary.exists() or (
        args.two_dgs_ply is not None
        and args.surface_table_output is not None
        and args.surface_table_output.exists()
    ):
        raise FileExistsError("planar map outputs must be new")
    started = time.perf_counter()
    if args.two_dgs_ply is not None:
        if args.surface_table_output is None:
            raise ValueError("--surface_table_output is required with --two_dgs_ply")
        table = _primitive_table_from_2dgs_ply(
            args.two_dgs_ply,
            minimum_opacity=float(args.surface_min_opacity),
            maximum_scale=args.surface_max_scale,
        )
        args.surface_table_output.parent.mkdir(parents=True, exist_ok=True)
        table.save_npz(args.surface_table_output)
        table = PrimitiveSurfaceTable.load_npz(args.surface_table_output)
        source_path = args.two_dgs_ply
        source_type = "raw_2dgs_ply"
    elif args.surface_table is not None:
        table = PrimitiveSurfaceTable.load_npz(args.surface_table)
        source_path = args.surface_table
        source_type = "direct_primitive_surface_table"
    else:
        physical = GoalMapletPhysicalMap.load_npz(args.physical_map)
        table = PrimitiveSurfaceTable.from_physical_map(physical)
        source_path = args.physical_map
        source_type = "legacy_wrapped_physical_map"
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
    planar = replace(planar, metadata={
        **planar.metadata,
        "primitive_geometry_source_type": source_type,
        "primitive_geometry_source_file_sha256": file_sha256(source_path),
        "primitive_surface_table_file_sha256": (
            file_sha256(args.surface_table_output)
            if args.surface_table_output is not None else None
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    planar.save_npz(args.output)
    loaded = type(planar).load_npz(args.output, primitive_count=table.primitive_ids.size)
    order = loaded.support_area_m2.argsort()[::-1]
    summary = {
        "artifact_type": "goal_maplet_geometry_native_planar_map_build_v1",
        "primitive_geometry_source_type": source_type,
        "primitive_geometry_source": str(source_path.resolve()),
        "primitive_geometry_source_file_sha256": file_sha256(source_path),
        "surface_table": (
            str(args.surface_table_output.resolve())
            if args.surface_table_output is not None else None
        ),
        "surface_table_file_sha256": (
            file_sha256(args.surface_table_output)
            if args.surface_table_output is not None else None
        ),
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
