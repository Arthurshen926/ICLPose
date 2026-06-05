"""Camera-view visualization for Stage F SfM vs VFM-native anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.rendered_map_verifier import project_xyz_to_image
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _load_track_stats(track_observations: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    reproj_by_track: dict[int, list[float]] = {}
    for obs in load_colmap_track_observations_jsonl(Path(track_observations)):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        reproj_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    return xyz_by_track, {int(track_id): float(np.mean(values)) for track_id, values in reproj_by_track.items() if values}


def _read_image_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for Stage F visualization") from exc
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for Stage F visualization") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _safe_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _project_index(index: LandmarkMapIndex, pose_w2c: np.ndarray, camera, max_points: int, rng: np.random.Generator):
    points = []
    order = np.arange(len(index), dtype=np.int64)
    if int(max_points) > 0 and order.size > int(max_points):
        order = np.sort(rng.choice(order, size=int(max_points), replace=False))
    for idx in order.tolist():
        xy = project_xyz_to_image(index.xyz[int(idx)], pose_w2c, camera)
        if xy is None:
            continue
        x, y = float(xy[0]), float(xy[1])
        if 0.0 <= x <= float(camera.width - 1) and 0.0 <= y <= float(camera.height - 1):
            points.append((x, y))
    return points


def _draw_points(image: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int], radius: int) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for Stage F visualization") from exc
    for x, y in points:
        cv2.circle(image, (int(round(x)), int(round(y))), int(radius), color, thickness=-1, lineType=cv2.LINE_AA)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize Stage F VFM-native anchors in query camera view")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--sfm_bank", required=True)
    parser.add_argument("--sfm_track_observations", required=True)
    parser.add_argument("--vfm_bank", required=True)
    parser.add_argument("--vfm_track_observations", required=True)
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--max_points_per_source", type=int, default=4000)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    poses = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    requested = {item.strip() for item in str(args.query_ids).split(",") if item.strip()}
    records = [record for record in manifest.records if record.image_id in poses and (not requested or record.image_id in requested)]
    if not requested and int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))

    sfm_xyz, sfm_reproj = _load_track_stats(Path(args.sfm_track_observations))
    vfm_xyz, vfm_reproj = _load_track_stats(Path(args.vfm_track_observations))
    sfm_index = LandmarkMapIndex.from_track_bank(load_selected_track_bank_npz(Path(args.sfm_bank)), sfm_xyz, sfm_reproj)
    vfm_index = LandmarkMapIndex.from_track_bank(load_selected_track_bank_npz(Path(args.vfm_bank)), vfm_xyz, vfm_reproj)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)
    rows = []
    rng = np.random.default_rng(0)
    for record in records:
        image = _read_image_rgb(image_root / record.image_id)
        sfm_points = _project_index(sfm_index, poses[record.image_id], camera, int(args.max_points_per_source), rng)
        vfm_points = _project_index(vfm_index, poses[record.image_id], camera, int(args.max_points_per_source), rng)
        overlay = np.asarray(image, dtype=np.uint8).copy()
        _draw_points(overlay, sfm_points, color=(0, 180, 255), radius=int(args.point_radius))
        _draw_points(overlay, vfm_points, color=(255, 70, 40), radius=int(args.point_radius))
        output_png = output_dir / f"{_safe_name(record.image_id)}_sfm_orange_vfm_blue.png"
        _write_image_rgb(output_png, overlay)
        rows.append(
            {
                "query_id": record.image_id,
                "output_png": str(output_png),
                "sfm_visible_points": int(len(sfm_points)),
                "vfm_visible_points": int(len(vfm_points)),
                "image_path": str(image_root / record.image_id),
            }
        )

    summary = {
        "stage": "stage_f_camera_view_anchor_visualization",
        "camera_source": camera_source,
        "query_count": int(len(rows)),
        "sfm_anchor_count": int(len(sfm_index)),
        "vfm_anchor_count": int(len(vfm_index)),
        "inputs": {
            "query_manifest": args.query_manifest,
            "query_pose_file": args.query_pose_file,
            "image_root": args.image_root,
            "sfm_bank": args.sfm_bank,
            "sfm_track_observations": args.sfm_track_observations,
            "vfm_bank": args.vfm_bank,
            "vfm_track_observations": args.vfm_track_observations,
        },
        "outputs": rows,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
