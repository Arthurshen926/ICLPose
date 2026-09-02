"""Seal source/held-disjoint MASt3R point-map inputs for a chart gate.

This audit is intentionally narrower than a reconstruction quality metric.  It
proves that the two MASt3R runs consumed disjoint, explicitly enumerated mapping
images and that both runs replay the frozen calibrated COLMAP cameras.  It never
opens Cambridge query pose text files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pycolmap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tree_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "sha256": _sha256(path),
            }
        )
    return rows


def _expected_names(images_dir: Path, indices: list[int]) -> list[str]:
    names = sorted(x.name for x in images_dir.iterdir() if x.is_file())
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("image indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= len(names):
        raise ValueError("image index is outside the posed COLMAP inventory")
    return [names[index] for index in indices]


def _official_camera_rows(posed_colmap: Path) -> dict[str, tuple[np.ndarray, float]]:
    reconstruction = pycolmap.Reconstruction(str(posed_colmap / "sparse" / "0"))
    rows = {}
    for image in reconstruction.images.values():
        matrix = np.asarray(image.cam_from_world().matrix(), np.float64)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -rotation.T @ translation
        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name != "PINHOLE" or camera.width != 1024 or camera.height != 576:
            raise ValueError("expected frozen 1024x576 PINHOLE camera")
        fx, fy, cx, cy = map(float, camera.params)
        if fx != fy or cx != 512.0 or cy != 288.0:
            raise ValueError("posed COLMAP camera is not the frozen ideal pinhole")
        rows[image.name] = (c2w, float(np.float32(0.5 * fx)))
    return rows


def _validate_run(
    root: Path,
    expected: list[str],
    allowed_routes: set[str],
    official: dict[str, tuple[np.ndarray, float]],
) -> dict:
    required = [root / "cameras.json", root / "pointmaps"]
    if not all(path.exists() for path in required):
        raise ValueError(f"incomplete MASt3R run: {root}")
    cameras = json.loads((root / "cameras.json").read_text())
    names = [Path(path).name for path in cameras["filepaths"]]
    if names != expected or len(set(names)) != len(names):
        raise ValueError("MASt3R camera order/inventory differs from frozen image_idx")
    routes = {name.split("__", 1)[0] for name in names}
    if routes != allowed_routes:
        raise ValueError("MASt3R run route inventory differs from its role")
    pointmaps = sorted(path.stem + ".png" for path in (root / "pointmaps").glob("*.json"))
    if pointmaps != sorted(names):
        raise ValueError("point-map inventory differs from camera inventory")
    images = sorted(path.name for path in (root / "images").glob("*.png"))
    if images != sorted(names):
        raise ValueError("copied image inventory differs from camera inventory")
    poses = np.asarray(cameras["cams2world"], np.float64)
    focals = np.asarray(cameras["focals"], np.float64)
    if poses.shape != (len(names), 4, 4) or focals.shape != (len(names),):
        raise ValueError("invalid calibrated camera arrays")
    pose_error = []
    focal_error = []
    for row, name in enumerate(names):
        if name not in official:
            raise ValueError("MASt3R camera is absent from posed COLMAP")
        target_pose, target_focal = official[name]
        pose_error.append(float(np.max(np.abs(poses[row] - target_pose))))
        focal_error.append(float(abs(focals[row] - target_focal)))
    if max(pose_error) > 1e-10 or max(focal_error) != 0.0:
        raise ValueError("MASt3R output does not replay frozen pose/intrinsics")
    pointmap_rows = [
        {
            "name": name,
            "file_sha256": _sha256(root / "pointmaps" / f"{Path(name).stem}.json"),
        }
        for name in sorted(names)
    ]
    return {
        "root": str(root.resolve()),
        "routes": sorted(routes),
        "image_count": len(names),
        "ordered_names": names,
        "cameras_file_sha256": _sha256(root / "cameras.json"),
        "pointmap_inventory_sha256": _canonical_sha256(pointmap_rows),
        "max_pose_replay_abs_error": max(pose_error),
        "max_focal_replay_abs_error": max(focal_error),
        "tree_sha256": _canonical_sha256(_tree_rows(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posed_colmap", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--held_root", type=Path, required=True)
    parser.add_argument("--source_indices", type=int, nargs="+", required=True)
    parser.add_argument("--held_indices", type=int, nargs="+", required=True)
    parser.add_argument("--source_routes", nargs="+", required=True)
    parser.add_argument("--held_routes", nargs="+", required=True)
    parser.add_argument("--forbidden_routes", nargs="+", default=["seq12", "seq14"])
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--isolated_inputs", type=Path)
    parser.add_argument("--preexecution_contract", type=Path)
    parser.add_argument("--pose_only_densification_plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite disjoint upstream authority")
    isolated = None
    preexecution = None
    if (args.isolated_inputs is None) != (args.preexecution_contract is None):
        raise ValueError("isolated inputs and preexecution contract must be supplied together")
    if args.isolated_inputs is not None:
        isolated = json.loads(args.isolated_inputs.read_text())
        isolated_claimed = isolated.pop("content_sha256", None)
        if isolated_claimed != _canonical_sha256(isolated):
            raise ValueError("isolated input manifest content hash differs")
        isolated["content_sha256"] = isolated_claimed
        preexecution = json.loads(args.preexecution_contract.read_text())
        preexecution_claimed = preexecution.pop("content_sha256", None)
        if preexecution_claimed != _canonical_sha256(preexecution):
            raise ValueError("preexecution contract content hash differs")
        preexecution["content_sha256"] = preexecution_claimed
        if (
            isolated.get("artifact_type")
            != "goal_maplet_isolated_chart_sfm_inputs_v1"
            or preexecution.get("artifact_type")
            != "goal_maplet_chart_sfm_preexecution_contract_v2"
            or preexecution.get("all_isolated_images_explicitly_indexed") is not True
            or preexecution["isolated_inputs_file_sha256"]
            != _sha256(args.isolated_inputs)
            or preexecution["isolated_inputs_content_sha256"] != isolated_claimed
            or Path(preexecution["source_command"][preexecution["source_command"].index("--output_dir") + 1]).resolve()
            != args.source_root.resolve()
            or Path(preexecution["held_command"][preexecution["held_command"].index("--output_dir") + 1]).resolve()
            != args.held_root.resolve()
        ):
            raise ValueError("isolated/preexecution binding differs")
        for role in ("source", "held"):
            command = preexecution[f"{role}_command"]
            image_count = int(isolated[role]["image_count"])
            if "--image_idx" not in command:
                raise ValueError("preexecution command does not enumerate all images")
            image_offset = command.index("--image_idx") + 1
            if command[image_offset:] != [str(index) for index in range(image_count)]:
                raise ValueError("preexecution image_idx is not the complete lexical inventory")
            scene = Path(command[command.index("--scene_path") + 1]).resolve()
            if scene != Path(isolated[role]["root"]).resolve():
                raise ValueError("preexecution scene root differs from isolated role")
    source_routes = set(args.source_routes)
    held_routes = set(args.held_routes)
    forbidden = set(args.forbidden_routes)
    if source_routes & held_routes or (source_routes | held_routes) & forbidden:
        raise ValueError("source/held/forbidden route sets are not disjoint")
    images_dir = args.posed_colmap / "images"
    source_names = _expected_names(images_dir, args.source_indices)
    held_names = _expected_names(images_dir, args.held_indices)
    densification = None
    if args.pose_only_densification_plan is not None:
        densification = json.loads(args.pose_only_densification_plan.read_text())
        densification_claimed = densification.pop("content_sha256", None)
        if densification_claimed != _canonical_sha256(densification):
            raise ValueError("pose-only densification plan content hash differs")
        densification["content_sha256"] = densification_claimed
        materialized_pose_routes = set(
            densification.get("routes_whose_camera_pose_fields_were_materialized", [])
        )
        if (
            densification.get("artifact_type")
            != "goal_maplet_pose_only_chart_densification_plan_v2"
            or densification.get("source_indices") != args.source_indices
            or densification.get("held_indices") != args.held_indices
            or densification.get("source_ordered_names") != source_names
            or densification.get("held_ordered_names") != held_names
            or densification.get("held_camera_fields_used_by_source_window_ranker")
            is not False
            or densification.get("other_route_camera_pose_fields_materialized")
            is not False
            or densification.get("query_or_forbidden_route_pose_fields_used")
            is not False
            or densification.get("points2D_or_point3D_fields_decoded") is not False
            or not source_routes.issubset(materialized_pose_routes)
            or not held_routes.issubset(materialized_pose_routes)
            or bool(materialized_pose_routes & forbidden)
            or Path(densification.get("posed_colmap_root", "")).resolve()
            != args.posed_colmap.resolve()
            or densification.get("posed_colmap_images_file_sha256")
            != _sha256(args.posed_colmap / "sparse" / "0" / "images.bin")
        ):
            raise ValueError("pose-only densification plan differs from isolated run")
    if isolated is not None:
        if (
            isolated["source"]["ordered_names"] != source_names
            or isolated["held"]["ordered_names"] != held_names
            or set(isolated["source"]["routes"]) != source_routes
            or set(isolated["held"]["routes"]) != held_routes
        ):
            raise ValueError("isolated input inventory differs from requested roles")
    if set(source_names) & set(held_names):
        raise ValueError("source and held images overlap")
    if {name.split("__", 1)[0] for name in source_names} != source_routes:
        raise ValueError("source image_idx does not match declared routes")
    if {name.split("__", 1)[0] for name in held_names} != held_routes:
        raise ValueError("held image_idx does not match declared routes")
    official = _official_camera_rows(args.posed_colmap)
    source = _validate_run(args.source_root, source_names, source_routes, official)
    held = _validate_run(args.held_root, held_names, held_routes, official)
    source_files = {
        "mast3r/run_mast3r.py": args.matcha_repo / "mast3r" / "run_mast3r.py",
        "mast3r/mast3r/cloud_opt/sparse_ga.py": args.matcha_repo
        / "mast3r"
        / "mast3r"
        / "cloud_opt"
        / "sparse_ga.py",
        "metric_checkpoint": args.matcha_repo
        / "mast3r"
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "retrieval_checkpoint": args.matcha_repo
        / "mast3r"
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    }
    observed_source_hashes = {name: _sha256(path) for name, path in source_files.items()}
    if preexecution is not None and preexecution["source_file_sha256"] != observed_source_hashes:
        raise ValueError("MASt3R source/checkpoint bytes changed after preexecution freeze")
    report = {
        "artifact_type": (
            "goal_maplet_disjoint_chart_upstream_authority_v2"
            if isolated is not None
            else "goal_maplet_disjoint_chart_upstream_authority_v1"
        ),
        "posed_colmap_root": str(args.posed_colmap.resolve()),
        "posed_colmap_cameras_file_sha256": _sha256(
            args.posed_colmap / "sparse" / "0" / "cameras.bin"
        ),
        "posed_colmap_images_file_sha256": _sha256(
            args.posed_colmap / "sparse" / "0" / "images.bin"
        ),
        "source_indices": args.source_indices,
        "held_indices": args.held_indices,
        "source": source,
        "held": held,
        "source_held_image_disjoint": True,
        "source_held_route_disjoint": True,
        "forbidden_routes": sorted(forbidden),
        "forbidden_routes_opened": False,
        "uses_mapping_camera_pose": True,
        "uses_query_or_ground_truth": False,
        "upstream_source_file_sha256": observed_source_hashes,
        "isolated_inputs_file_sha256": (
            _sha256(args.isolated_inputs) if args.isolated_inputs else None
        ),
        "isolated_inputs_content_sha256": (
            isolated["content_sha256"] if isolated else None
        ),
        "isolated_source_input": (
            {
                key: isolated["source"][key]
                for key in (
                    "root",
                    "image_count",
                    "ordered_names",
                    "cameras_file_sha256",
                    "images_file_sha256",
                    "points3D_file_sha256",
                )
            }
            if isolated
            else None
        ),
        "isolated_held_input": (
            {
                key: isolated["held"][key]
                for key in (
                    "root",
                    "image_count",
                    "ordered_names",
                    "cameras_file_sha256",
                    "images_file_sha256",
                    "points3D_file_sha256",
                )
            }
            if isolated
            else None
        ),
        "preexecution_contract_file_sha256": (
            _sha256(args.preexecution_contract) if args.preexecution_contract else None
        ),
        "preexecution_contract_content_sha256": (
            preexecution["content_sha256"] if preexecution else None
        ),
        "pose_only_densification_plan_file_sha256": (
            _sha256(args.pose_only_densification_plan)
            if args.pose_only_densification_plan
            else None
        ),
        "pose_only_densification_plan_content_sha256": (
            densification["content_sha256"] if densification else None
        ),
        "physical_source_held_input_roots_disjoint": bool(isolated),
        "strict_disjoint_upstream": True,
        "production_eligible": False,
        "diagnostic_semantics": "source/held-disjoint_mapping_geometry_gate_not_sensor_depth_GT",
    }
    report["content_sha256"] = _canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in {"source", "held"}}, indent=2))


if __name__ == "__main__":
    main()
