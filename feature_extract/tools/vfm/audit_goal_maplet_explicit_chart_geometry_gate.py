"""Seal the bounded explicit-chart geometry comparison and its source lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dav2_dir", type=Path, required=True)
    parser.add_argument("--moge3_dir", type=Path, required=True)
    parser.add_argument("--direct_report", type=Path, required=True)
    parser.add_argument("--physical_map", type=Path, required=True)
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite chart gate audit")
    dav2_manifest_path = args.dav2_dir / "manifest.json"
    moge_manifest_path = args.moge3_dir / "manifest.json"
    dav2_atlas_path = args.dav2_dir / "explicit_atlas_stride4.npz"
    moge_atlas_path = args.moge3_dir / "explicit_atlas_stride4.npz"
    dav2_atlas_report_path = dav2_atlas_path.with_suffix(".json")
    moge_atlas_report_path = moge_atlas_path.with_suffix(".json")
    dav2_held_path = args.dav2_dir / "held_seq4_triangle_reprojection_v2.json"
    moge_held_path = args.moge3_dir / "held_seq4_triangle_reprojection_v2.json"
    dav2_manifest = load(dav2_manifest_path)
    moge_manifest = load(moge_manifest_path)
    dav2_report = load(dav2_atlas_report_path)
    moge_report = load(moge_atlas_report_path)
    dav2_held = load(dav2_held_path)
    moge_held = load(moge_held_path)
    if dav2_manifest["chart_names"] != moge_manifest["chart_names"]:
        raise ValueError("M1/M2 chart inventories differ")
    if dav2_manifest["iterations"] != 1000 or moge_manifest["iterations"] != 1000:
        raise ValueError("M1/M2 do not both use the frozen 1000-iteration gate")
    for path, report in ((dav2_atlas_path, dav2_report), (moge_atlas_path, moge_report)):
        if sha(path) != report["output_file_sha256"]:
            raise ValueError("atlas bytes differ from report")
    for path, report in ((dav2_atlas_path, dav2_held), (moge_atlas_path, moge_held)):
        if sha(path) != report["atlas_file_sha256"]:
            raise ValueError("held-view report binds another atlas")
    sources = [
        Path("/root/ICLPose/feature_extract/vfm/localization_goal_maplet/explicit_chart_atlas.py"),
        Path("/root/ICLPose/feature_extract/tools/vfm/build_goal_maplet_explicit_chart_atlas.py"),
        Path("/root/ICLPose/feature_extract/tools/vfm/build_goal_maplet_moge3_chart_initializers.py"),
        Path("/root/ICLPose/feature_extract/tools/vfm/run_goal_maplet_masked_chart_alignment_gate.py"),
        Path("/root/ICLPose/feature_extract/tools/vfm/evaluate_goal_maplet_chart_held_reprojection.py"),
        args.matcha_repo / "matcha/dm_scene/parallel_aligner.py",
        args.matcha_repo / "matcha/dm_modules/matcher_3d.py",
    ]
    result = {
        "artifact_type": "goal_maplet_explicit_chart_geometry_gate_audit_v2",
        "chart_inventory_equal": True,
        "chart_count": len(dav2_manifest["chart_names"]),
        "iterations": 1000,
        "route_clean": bool(dav2_manifest["route_clean"] and moge_manifest["route_clean"]),
        "uses_query_or_ground_truth": False,
        "source_file_sha256": {str(path): sha(path) for path in sources},
        "input_output_file_sha256": {
            str(path): sha(path) for path in (
                dav2_manifest_path, moge_manifest_path,
                dav2_atlas_path, moge_atlas_path,
                dav2_atlas_report_path, moge_atlas_report_path,
                dav2_held_path, moge_held_path,
                args.direct_report, args.physical_map,
            )
        },
        "physical_2dgs_map_mib": args.physical_map.stat().st_size / 2**20,
        "dav2": {
            "atlas_mib": dav2_report["npz_mib"],
            "cross_chart_median_m": dav2_report["aligned_cross_chart_nearest"]["median_m"],
            "cross_chart_p90_m": dav2_report["aligned_cross_chart_nearest"]["p90_m"],
            "within_0p5m": dav2_report["aligned_cross_chart_nearest"]["within_0p5m"],
            **dav2_report["aligned_surface_consistency"],
            "held_coverage": dav2_held["coverage"],
            "held_depth_median_m": dav2_held["absolute_depth_median_m"],
            "held_depth_p90_m": dav2_held["absolute_depth_p90_m"],
            "held_relative_depth_median": dav2_held["relative_depth_median"],
            "held_relative_depth_p90": dav2_held["relative_depth_p90"],
        },
        "moge3": {
            "atlas_mib": moge_report["npz_mib"],
            "cross_chart_median_m": moge_report["aligned_cross_chart_nearest"]["median_m"],
            "cross_chart_p90_m": moge_report["aligned_cross_chart_nearest"]["p90_m"],
            "within_0p5m": moge_report["aligned_cross_chart_nearest"]["within_0p5m"],
            **moge_report["aligned_surface_consistency"],
            "held_coverage": moge_held["coverage"],
            "held_depth_median_m": moge_held["absolute_depth_median_m"],
            "held_depth_p90_m": moge_held["absolute_depth_p90_m"],
            "held_relative_depth_median": moge_held["relative_depth_median"],
            "held_relative_depth_p90": moge_held["relative_depth_p90"],
        },
        "decision": {
            "explicit_chart_representation": "GO_TO_FULL_SUBMAP_GEOMETRY_GATE",
            "moge3_initializer": "GO_TO_FULL_SUBMAP_GATE_BUT_NOT_DOMINANT_OVER_DAV2",
            "direct_moge3_fusion": "KILL_AS_FINAL_MAP",
            "replace_2dgs_now": "KILL",
            "run_pose_backend_now": "KILL",
        },
        "blockers": [
            "only_eight_seq4_charts",
            "no_chart_family_canonicalization",
            "no_RADIO_UV_field",
            "held_reference_is_MASt3R_not_sensor_depth_GT",
            "cross_chart_distance_p90_remains_large",
        ],
    }
    result["content_sha256"] = canonical(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["content_sha256"])


if __name__ == "__main__":
    main()
