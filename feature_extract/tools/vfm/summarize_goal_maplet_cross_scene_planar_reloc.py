"""Seal a compact cross-scene PlanarReloc-style map/PnP validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256


def metrics(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text()); rows = report["rows"]
    translation = np.asarray([row["translation_error_m"] for row in rows], np.float64)
    rotation = np.asarray([row["rotation_error_deg"] for row in rows], np.float64)
    result = {
        "file_sha256": file_sha256(path), "query_count": len(rows),
        "median_translation_m": float(np.median(translation)),
        "p90_translation_m": float(np.quantile(translation, .9)),
        "maximum_translation_m": float(np.max(translation)),
        "median_rotation_deg": float(np.median(rotation)),
        "p90_rotation_deg": float(np.quantile(rotation, .9)),
        "maximum_rotation_deg": float(np.max(rotation)),
    }
    for distance, angle, name in ((.1, 1, "10cm1deg"), (.25, 2, "25cm2deg"),
                                  (.5, 5, "50cm5deg"), (1, 10, "1m10deg"),
                                  (2, 45, "2m45deg")):
        result[f"recall_{name}"] = float(np.mean((translation <= distance) & (rotation <= angle)))
    return result


def aggregate(paths: list[Path]) -> dict[str, object]:
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text())["rows"])
    translation = np.asarray([row["translation_error_m"] for row in rows], np.float64)
    rotation = np.asarray([row["rotation_error_deg"] for row in rows], np.float64)
    return {
        "query_count": len(rows),
        "median_translation_m": float(np.median(translation)),
        "median_rotation_deg": float(np.median(rotation)),
        "recall_25cm2deg": float(np.mean((translation <= .25) & (rotation <= 2))),
        "recall_50cm5deg": float(np.mean((translation <= .5) & (rotation <= 5))),
        "recall_1m10deg": float(np.mean((translation <= 1) & (rotation <= 10))),
        "recall_2m45deg": float(np.mean((translation <= 2) & (rotation <= 45))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_report", type=Path, required=True)
    parser.add_argument("--seq1_planes", type=Path, required=True)
    parser.add_argument("--seq3_planes", type=Path, required=True)
    parser.add_argument("--seq1_top1", type=Path, required=True)
    parser.add_argument("--seq1_top5", type=Path, required=True)
    parser.add_argument("--seq3_top1", type=Path, required=True)
    parser.add_argument("--seq3_top5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite cross-scene summary")
    map_report = json.loads(args.map_report.read_text())
    routes = {}
    for route, plane_path, top1, top5 in (
        ("seq1", args.seq1_planes, args.seq1_top1, args.seq1_top5),
        ("seq3", args.seq3_planes, args.seq3_top1, args.seq3_top5),
    ):
        plane = json.loads(plane_path.read_text())
        routes[route] = {
            "query_plane_manifest_file_sha256": file_sha256(plane_path),
            "query_plane_count_mean": plane["plane_count"] / plane["query_count"],
            "query_plane_pixel_coverage_fraction": (
                plane["covered_pixel_count"] / (plane["query_count"] * 144 * 256)
            ),
            "top1": metrics(top1), "top5": metrics(top5),
        }
    payload = {
        "artifact_type": "goal_maplet_cross_scene_planarreloc_validation_v1",
        "scene": "ShopFacade", "mapping_route": "seq2", "query_routes": ["seq1", "seq3"],
        "map": {
            "report_file_sha256": file_sha256(args.map_report),
            "mapping_view_count": map_report["view_count"],
            "finite_plane_count": map_report["plane_count"],
            "minimum_mapping_views_per_plane": map_report["minimum_views"],
            "plane_count_area_ge_1m2": map_report["area_ge_1m2"],
            "plane_count_area_ge_5m2": map_report["area_ge_5m2"],
            "plane_count_area_ge_10m2": map_report["area_ge_10m2"],
            "median_plane_area_m2": map_report["median_area_m2"],
            "median_plane_fit_rms_m": map_report["median_rms_m"],
        },
        "routes": routes,
        "aggregate": {
            "top1": aggregate([args.seq1_top1, args.seq3_top1]),
            "top5": aggregate([args.seq1_top5, args.seq3_top5]),
        },
        "contracts": {
            "query_moge3_role": "plane_segmentation_only",
            "query_depth_or_scale_used_by_pose_solver": False,
            "query_camera_authority_pose_or_ground_truth_file_parsed": False,
            "mapping_and_query_routes_disjoint": True,
            "radio_token_grid": [68, 120],
            "post_shopfacade_parameter_tuning": False,
            "production_eligible": False,
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "file_sha256": file_sha256(args.output),
                      "content_sha256": payload["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
