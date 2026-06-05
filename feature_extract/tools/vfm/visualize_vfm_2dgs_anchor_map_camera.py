"""Visualize VFM-2DGS anchors from posed camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
    _parse_default_camera,
    _select_records,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.rendered_map_verifier import project_xyz_to_image
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap, Vfm2DgsObservationBank


def _safe_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _read_image_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _draw_point(image: np.ndarray, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    cv2.circle(image, (int(round(x)), int(round(y))), int(radius), color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_box(image: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    cv2.rectangle(
        image,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _parse_grid(value: str) -> tuple[int, int] | None:
    text = str(value).strip()
    if not text:
        return None
    if "x" in text:
        left, right = text.lower().split("x", 1)
    elif "," in text:
        left, right = text.split(",", 1)
    else:
        raise ValueError("token grid must be HxW or H,W")
    height, width = int(left), int(right)
    if height <= 0 or width <= 0:
        raise ValueError("token grid dimensions must be positive")
    return height, width


def _scale_from_camera_to_image(image_rgb: np.ndarray, camera) -> tuple[float, float]:
    height, width = np.asarray(image_rgb).shape[:2]
    return (
        float(width) / max(float(camera.width), 1.0),
        float(height) / max(float(camera.height), 1.0),
    )


def _quality_color(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    denom = max(float(vmax) - float(vmin), 1e-8)
    t = float(np.clip((float(value) - float(vmin)) / denom, 0.0, 1.0))
    return (int(60 + 195 * t), int(220 - 120 * t), int(255 - 220 * t))


def _project_visible_points(xyz: np.ndarray, pose_w2c: np.ndarray, camera) -> list[tuple[float, float]]:
    points = []
    for point in np.asarray(xyz, dtype=np.float64).reshape(-1, 3):
        xy = project_xyz_to_image(point, pose_w2c, camera)
        if xy is None:
            continue
        x, y = float(xy[0]), float(xy[1])
        if 0.0 <= x <= float(camera.width - 1) and 0.0 <= y <= float(camera.height - 1):
            points.append((x, y))
    return points


def _project_visible_indexed(xyz: np.ndarray, pose_w2c: np.ndarray, camera) -> list[tuple[int, float, float]]:
    points = []
    for idx, point in enumerate(np.asarray(xyz, dtype=np.float64).reshape(-1, 3)):
        xy = project_xyz_to_image(point, pose_w2c, camera)
        if xy is None:
            continue
        x, y = float(xy[0]), float(xy[1])
        if 0.0 <= x <= float(camera.width - 1) and 0.0 <= y <= float(camera.height - 1):
            points.append((int(idx), x, y))
    return points


def _support_xyz(anchor_map: Vfm2DgsAnchorMap, gaussian_ply: str, max_support_points: int) -> np.ndarray:
    if not gaussian_ply:
        return np.zeros((0, 3), dtype=np.float64)
    source = load_gaussian_vfm_source_from_ply(Path(gaussian_ply), max_gaussians=0)
    row_by_id = {
        int(gaussian_index): int(row)
        for row, gaussian_index in enumerate(np.asarray(source.gaussian_indices, dtype=np.int64).tolist())
    }
    support_parent_ids = getattr(anchor_map, "support_parent_gaussian_indices", anchor_map.support_element_ids)
    rows = [
        row_by_id[int(element_id)]
        for element_id in np.unique(support_parent_ids).tolist()
        if int(element_id) in row_by_id
    ]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    rows_arr = np.asarray(rows, dtype=np.int64)
    if int(max_support_points) > 0 and rows_arr.size > int(max_support_points):
        rng = np.random.default_rng(0)
        rows_arr = np.sort(rng.choice(rows_arr, size=int(max_support_points), replace=False))
    return np.asarray(source.xyz, dtype=np.float64)[rows_arr]


def _surface_support_xyz(anchor_map: Vfm2DgsAnchorMap, surface_npz: str, max_support_points: int) -> np.ndarray:
    if not surface_npz:
        return np.zeros((0, 3), dtype=np.float64)
    elements = SurfaceElementMap.load_npz(Path(surface_npz))
    row_by_id = {int(element_id): int(row) for row, element_id in enumerate(elements.element_ids.tolist())}
    rows = [
        row_by_id[int(element_id)]
        for element_id in np.unique(anchor_map.support_element_ids).tolist()
        if int(element_id) in row_by_id
    ]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    rows_arr = np.asarray(rows, dtype=np.int64)
    if int(max_support_points) > 0 and rows_arr.size > int(max_support_points):
        rng = np.random.default_rng(0)
        rows_arr = np.sort(rng.choice(rows_arr, size=int(max_support_points), replace=False))
    return np.asarray(elements.centers, dtype=np.float64)[rows_arr]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize VFM-2DGS anchors in camera views")
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--gaussian_ply", default="")
    parser.add_argument("--surface_npz", default="")
    parser.add_argument("--observation_bank", default="")
    parser.add_argument("--token_grid", default="")
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--max_support_points", type=int, default=12000)
    parser.add_argument("--anchor_radius", type=int, default=4)
    parser.add_argument("--support_radius", type=int, default=1)
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    anchor_map = Vfm2DgsAnchorMap.load_npz(Path(args.anchor_map))
    observation_bank = Vfm2DgsObservationBank.load_npz(Path(args.observation_bank)) if args.observation_bank else None
    token_grid = _parse_grid(args.token_grid)
    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    poses = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    cameras = _load_camera_by_image(args.camera_model_dir)
    fallback_camera = _parse_default_camera(args.default_camera)
    requested = {item.strip() for item in str(args.query_ids).split(",") if item.strip()}
    records = [record for record in manifest.records if record.image_id in poses and (not requested or record.image_id in requested)]
    if not requested and int(args.max_queries) > 0:
        records = _select_records(records, int(args.max_queries), "uniform")

    support_xyz = (
        _surface_support_xyz(anchor_map, args.surface_npz, int(args.max_support_points))
        if args.surface_npz
        else _support_xyz(anchor_map, args.gaussian_ply, int(args.max_support_points))
    )
    quality = np.asarray(anchor_map.quality_scores, dtype=np.float32)
    qmin = float(np.percentile(quality, 5.0)) if quality.size else 0.0
    qmax = float(np.percentile(quality, 95.0)) if quality.size else 1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        pose = poses[record.image_id]
        camera = cameras.get(record.image_id, fallback_camera)
        image = _read_image_rgb(Path(args.image_root) / record.image_id)
        scale_x, scale_y = _scale_from_camera_to_image(image, camera)
        overlay = np.asarray(image, dtype=np.uint8).copy()
        support_points = _project_visible_points(support_xyz, pose, camera)
        for x, y in support_points:
            _draw_point(overlay, x * scale_x, y * scale_y, color=(60, 210, 255), radius=int(args.support_radius))
        anchor_points = []
        for idx, x, y in _project_visible_indexed(anchor_map.centers, pose, camera):
            color = _quality_color(float(anchor_map.quality_scores[idx]), qmin, qmax)
            radius = max(int(args.anchor_radius), int(np.sqrt(max(float(anchor_map.surface_support_counts[idx]), 1.0))))
            _draw_point(overlay, x * scale_x, y * scale_y, color=color, radius=radius)
            anchor_points.append((x * scale_x, y * scale_y))
        observation_count = 0
        if observation_bank is not None and token_grid is not None:
            grid_h, grid_w = token_grid
            image_h, image_w = image.shape[:2]
            for obs_row, image_id in enumerate(observation_bank.image_ids):
                if str(image_id) != str(record.image_id):
                    continue
                token_x, token_y = observation_bank.token_xy[obs_row]
                cx = (float(token_x) + 0.5) * float(image_w) / float(grid_w)
                cy = (float(token_y) + 0.5) * float(image_h) / float(grid_h)
                half_w = 0.5 * float(image_w) / float(grid_w)
                half_h = 0.5 * float(image_h) / float(grid_h)
                _draw_box(overlay, cx - half_w, cy - half_h, cx + half_w, cy + half_h, color=(255, 255, 40))
                observation_count += 1
        output_png = output_dir / f"{_safe_name(record.image_id)}_vfm_2dgs_anchor_map.png"
        _write_image_rgb(output_png, overlay)
        rows.append(
            {
                "image_id": record.image_id,
                "output_png": str(output_png),
                "visible_anchor_count": int(len(anchor_points)),
                "visible_support_element_count": int(len(support_points)),
                "token_observation_count": int(observation_count),
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "camera_width": int(camera.width),
                "camera_height": int(camera.height),
                "camera_to_image_scale": [float(scale_x), float(scale_y)],
            }
        )

    summary = {
        "stage": "vfm_2dgs_anchor_camera_visualization",
        "anchor_count": int(len(anchor_map)),
        "support_element_count": int(anchor_map.support_element_ids.size),
        "visualized_query_count": int(len(rows)),
        "outputs": rows,
        "inputs": {
            "anchor_map": args.anchor_map,
            "reference_manifest": args.reference_manifest,
            "reference_pose_file": args.reference_pose_file,
            "image_root": args.image_root,
            "gaussian_ply": args.gaussian_ply,
            "surface_npz": args.surface_npz,
            "observation_bank": args.observation_bank,
            "token_grid": args.token_grid,
        },
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
