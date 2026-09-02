"""Freeze a source-only local camera window for the next chart SfM run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_pose_only_densification import (
    MappingCameraPose,
    PoseOnlyDensificationConfig,
    SCHEMA,
    diagnose_held_camera_neighbors,
    ordered_camera_names_sha256,
    select_source_camera_window,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def _quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, np.float64)
    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid COLMAP image quaternion")
    w, x, y, z = qvec / norm
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _read_colmap_image_poses_only(
    path: Path,
    *,
    materialized_routes: set[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Read pose/name fields, materializing only explicitly requested routes.

    Every 2D/3D observation block is skipped with ``seek`` and never decoded.
    """

    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observed_names: list[str] = []
    with Path(path).open("rb") as stream:
        raw_count = stream.read(8)
        if len(raw_count) != 8:
            raise ValueError("truncated COLMAP images.bin")
        image_count = struct.unpack("<Q", raw_count)[0]
        for _ in range(image_count):
            raw_pose = stream.read(64)
            if len(raw_pose) != 64:
                raise ValueError("truncated COLMAP image pose")
            values = struct.unpack("<idddddddi", raw_pose)
            qvec = np.asarray(values[1:5], np.float64)
            tvec = np.asarray(values[5:8], np.float64)
            name_bytes = bytearray()
            while True:
                value = stream.read(1)
                if not value:
                    raise ValueError("unterminated COLMAP image name")
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            name = name_bytes.decode("utf-8")
            raw_points = stream.read(8)
            if len(raw_points) != 8:
                raise ValueError("truncated COLMAP point count")
            point_count = struct.unpack("<Q", raw_points)[0]
            stream.seek(24 * point_count, os.SEEK_CUR)
            if name in observed_names:
                raise ValueError("duplicate COLMAP image name")
            observed_names.append(name)
            if name.split("__", 1)[0] in materialized_routes:
                world_to_camera = _quaternion_to_rotation(qvec)
                center = -world_to_camera.T @ tvec
                forward = world_to_camera.T[:, 2]
                rows[name] = (center, forward)
        if stream.read(1):
            raise ValueError("COLMAP images.bin has trailing bytes")
    return rows, observed_names


def _trajectory_summary(cameras: list[MappingCameraPose]) -> dict[str, object]:
    centers = np.stack([row.center_world for row in cameras])
    forwards = np.stack([row.forward_world for row in cameras])
    adjacent_baseline = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    adjacent_angle = np.degrees(
        np.arccos(np.clip(np.sum(forwards[:-1] * forwards[1:], axis=1), -1.0, 1.0))
    )
    pairwise = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    pairwise = pairwise[np.triu_indices(len(cameras), 1)]
    path_length = float(adjacent_baseline.sum())
    displacement = float(np.linalg.norm(centers[-1] - centers[0]))
    return {
        "adjacent_baseline_m": {
            "minimum": float(adjacent_baseline.min()),
            "median": float(np.median(adjacent_baseline)),
            "maximum": float(adjacent_baseline.max()),
        },
        "adjacent_forward_angle_deg": {
            "minimum": float(adjacent_angle.min()),
            "median": float(np.median(adjacent_angle)),
            "maximum": float(adjacent_angle.max()),
        },
        "path_length_m": path_length,
        "endpoint_displacement_m": displacement,
        "endpoint_displacement_over_path": displacement / path_length,
        "all_pair_baseline_m": {
            "p10": float(np.quantile(pairwise, 0.1)),
            "median": float(np.median(pairwise)),
            "p90": float(np.quantile(pairwise, 0.9)),
            "maximum": float(pairwise.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posed_colmap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_route", default="seq4")
    parser.add_argument("--held_routes", nargs="+", default=["seq1", "seq9"])
    parser.add_argument("--source_view_count", type=int, default=24)
    parser.add_argument("--maximum_adjacent_baseline_m", type=float, default=1.5)
    parser.add_argument("--maximum_adjacent_forward_angle_deg", type=float, default=12.0)
    parser.add_argument("--minimum_path_length_m", type=float, default=15.0)
    parser.add_argument("--held_neighbor_radius_m", type=float, default=6.0)
    parser.add_argument("--held_neighbor_forward_angle_deg", type=float, default=45.0)
    parser.add_argument("--held_target_view_count_per_route", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite pose-only densification plan")
    if args.source_route in set(args.held_routes):
        raise ValueError("source and held routes must be disjoint")
    config = PoseOnlyDensificationConfig(
        source_view_count=args.source_view_count,
        maximum_adjacent_baseline_m=args.maximum_adjacent_baseline_m,
        maximum_adjacent_forward_angle_deg=args.maximum_adjacent_forward_angle_deg,
        minimum_path_length_m=args.minimum_path_length_m,
        held_neighbor_radius_m=args.held_neighbor_radius_m,
        held_neighbor_forward_angle_deg=args.held_neighbor_forward_angle_deg,
        held_target_view_count_per_route=args.held_target_view_count_per_route,
    ).validated()

    posed_colmap = args.posed_colmap.resolve()
    images_binary = posed_colmap / "sparse" / "0" / "images.bin"
    image_directory = posed_colmap / "images"
    lexical_names = sorted(path.name for path in image_directory.iterdir() if path.is_file())
    if not lexical_names or len(set(lexical_names)) != len(lexical_names):
        raise ValueError("posed COLMAP image-name inventory is empty or duplicated")
    materialized_routes = {args.source_route, *args.held_routes}
    pose_rows, observed_pose_names = _read_colmap_image_poses_only(
        images_binary,
        materialized_routes=materialized_routes,
    )
    if set(observed_pose_names) != set(lexical_names):
        raise ValueError("COLMAP pose and image-name inventories differ")
    global_index = {name: row for row, name in enumerate(lexical_names)}
    by_route: dict[str, list[MappingCameraPose]] = {}
    for name, (center, forward) in pose_rows.items():
        route = name.split("__", 1)[0]
        by_route.setdefault(route, []).append(
            MappingCameraPose(
                name=name,
                global_lexical_index=global_index[name],
                center_world=center,
                forward_world=forward,
            ).validated()
        )
    if args.source_route not in by_route:
        raise ValueError("source route is absent from posed COLMAP")
    if any(route not in by_route for route in args.held_routes):
        raise ValueError("a held route is absent from posed COLMAP")

    # This call is intentionally completed before held-route rows are passed to
    # any function.  Source ranking cannot depend on held camera availability.
    selected_source, source_candidates = select_source_camera_window(
        by_route[args.source_route], config=config
    )
    held_diagnostics, selected_held = diagnose_held_camera_neighbors(
        selected_source,
        {route: by_route[route] for route in args.held_routes},
        config=config,
    )
    selected_held_routes = sorted({row.route for row in selected_held})
    selected_held_by_route = {
        route: [row for row in selected_held if row.route == route]
        for route in selected_held_routes
    }
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "posed_colmap_root": str(posed_colmap),
        "posed_colmap_images_file_sha256": file_sha256(images_binary),
        "lexical_image_name_inventory_sha256": canonical_json_sha256(lexical_names),
        "source_route": args.source_route,
        "held_routes_diagnosed_after_source_freeze": sorted(args.held_routes),
        "source_indices": [row.global_lexical_index for row in selected_source],
        "source_ordered_names": [row.name for row in selected_source],
        "source_ordered_names_sha256": ordered_camera_names_sha256(selected_source),
        "held_indices": [row.global_lexical_index for row in selected_held],
        "held_ordered_names": [row.name for row in selected_held],
        "held_ordered_names_sha256": ordered_camera_names_sha256(selected_held),
        "recommended_held_routes": selected_held_routes,
        "source_trajectory": _trajectory_summary(selected_source),
        "selected_held_trajectory_by_route": {
            route: _trajectory_summary(rows)
            for route, rows in selected_held_by_route.items()
            if len(rows) >= 2
        },
        "source_window_candidate_count": len(source_candidates),
        "source_window_gate_pass_count": sum(
            row["source_only_gate_pass"] for row in source_candidates
        ),
        "source_window_candidates": source_candidates,
        "held_post_selection_camera_neighbor_diagnostic": held_diagnostics,
        "config": config.to_dict(),
        "source_selection_rule": (
            "fixed local source-route window; gate adjacent baseline, adjacent forward angle, "
            "and path length; lexicographically maximize endpoint_displacement/path_length, "
            "then path_length, then earliest source start"
        ),
        "source_selection_numeric_fields": [
            "source_camera_center_world",
            "source_camera_forward_world",
            "source_lexical_order",
        ],
        "held_camera_fields_used_by_source_window_ranker": False,
        "held_camera_diagnostic_executed_only_after_source_freeze": True,
        "routes_whose_camera_pose_fields_were_materialized": sorted(materialized_routes),
        "other_route_camera_pose_fields_materialized": False,
        "points2D_or_point3D_fields_decoded": False,
        "uses_initializer_or_surface_geometry": False,
        "uses_rgb_numeric_fields": False,
        "uses_query_or_ground_truth": False,
        "query_or_forbidden_route_pose_fields_used": False,
        "expected_semantics": (
            "pose_only_candidate_for_new_physical_isolation_not_a_surface_overlap_proof"
        ),
        "decision": (
            "GO_freeze_source_and_available_seq1_held_camera_indices_for_isolated_v4_run"
            if selected_held
            else "KILL_no_independent_held_camera_neighbor_inventory"
        ),
        "limitations": [
            "camera proximity and forward angle do not prove shared visible surface",
            "seq9 has no nearby camera under the frozen post-selection gate if its eligible count is zero",
            "the new isolated reconstruction must still pass the model-neutral overlap selector",
        ],
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"source_window_candidates"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
