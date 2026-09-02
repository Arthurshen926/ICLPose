"""Build a pose-free fixed-calibration camera manifest for mapping images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def build_manifest(
    *,
    pose_file: Path,
    radio_manifest: Path,
    calibration_manifest: Path,
    routes: tuple[str, ...],
) -> dict[str, object]:
    radio = json.loads(radio_manifest.read_text())
    records = list(radio["records"])
    allowed = set(routes)
    names = sorted(
        str(row["image_id"])
        for row in records
        if str(row["image_id"]).split("/", 1)[0] in allowed
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("mapping RADIO inventory is empty or duplicated")
    poses = {row.image_id for row in parse_cambridge_pose_file(pose_file)}
    if not set(names).issubset(poses):
        raise ValueError("mapping RADIO images are absent from mapping pose authority")

    calibration = json.loads(calibration_manifest.read_text())
    rows = list(calibration["cameras"].values())
    if not rows or any(row != rows[0] for row in rows[1:]):
        raise ValueError("calibration authority is not fixed across mapping views")
    fixed = rows[0]
    camera = {
        "model_id": int(fixed["model_id"]),
        "width": int(fixed["width"]),
        "height": int(fixed["height"]),
        "params": [float(value) for value in fixed["params"]],
    }
    if len(camera["params"]) != 4:
        raise ValueError("fixed calibration parameter count differs")
    cameras = {name: dict(camera) for name in names}
    route_counts = {
        route: sum(name.split("/", 1)[0] == route for name in names)
        for route in sorted(allowed)
    }
    payload: dict[str, object] = {
        "format": "per_query_colmap_calibration_only_v1",
        "query_count": len(names),
        "cameras": cameras,
        "route_counts": route_counts,
        "production_contract": {
            "contains_camera_pose": False,
            "contains_sfm_points": False,
            "contains_sfm_tracks": False,
            "contains_query_or_ground_truth": False,
            "mapping_only": True,
        },
        "intrinsic_audit": {
            "source": "fixed_MAtCha_mapping_calibration_replayed_over_mapping_RADIO_inventory",
            "mapping_route_count": len(route_counts),
        },
        "lineage": {
            "mapping_pose_file_sha256": file_sha256(pose_file),
            "mapping_radio_manifest_file_sha256": file_sha256(radio_manifest),
            "fixed_calibration_manifest_file_sha256": file_sha256(calibration_manifest),
            "pose_values_materialized": False,
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_file", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--calibration_manifest", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite mapping camera manifest")
    payload = build_manifest(
        pose_file=args.pose_file,
        radio_manifest=args.radio_manifest,
        calibration_manifest=args.calibration_manifest,
        routes=tuple(args.routes),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "camera_count": payload["query_count"],
        "route_counts": payload["route_counts"],
        "content_sha256": payload["content_sha256"],
        "output_file_sha256": file_sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
