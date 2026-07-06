from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _safe_image_stem
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_images_binary
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


def project_world_points_to_image(
    points_xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with COLMAP intrinsics/distortion."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for D0 projection") from exc
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    projected, _ = cv2.projectPoints(points, rvec, pose[:3, 3], matrix, distortion)
    camera_xyz = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    xy = projected.reshape(-1, 2).astype(np.float64, copy=False)
    valid = np.isfinite(xy).all(axis=1) & (camera_xyz[:, 2] > 1e-8)
    valid &= (xy[:, 0] >= 0.0) & (xy[:, 0] < float(camera.width)) & (xy[:, 1] >= 0.0) & (xy[:, 1] < float(camera.height))
    return xy, valid


def camera_by_image_id_from_colmap(model_dir: Path, cameras: Mapping[int, ColmapCamera]) -> dict[str, ColmapCamera]:
    images = read_colmap_images_binary(Path(model_dir) / "images.bin")
    by_id: dict[str, ColmapCamera] = {}
    for image in images.values():
        if int(image.camera_id) in cameras:
            by_id[str(image.image_name)] = cameras[int(image.camera_id)]
    return by_id


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _float(row: dict[str, str], name: str) -> float:
    return float(str(row.get(name, "")).strip())


def _stride4_rgb_feature(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB input must have shape (H,W,3)")
    height, width = int(image.shape[0]), int(image.shape[1])
    out_w = max(width // 4, 1)
    out_h = max(height // 4, 1)
    resized = Image.fromarray(image).resize((out_w, out_h), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.moveaxis(arr, -1, 0).astype(np.float32, copy=False)


def _write_stride4_rgb_cache(path: Path, rgb: np.ndarray) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stride4_rgb=_stride4_rgb_feature(rgb))
    return path


def _render_cache_by_query(render_cache_manifest_csv: Path | None) -> dict[str, Path]:
    if render_cache_manifest_csv is None:
        return {}
    mapping: dict[str, Path] = {}
    for row in _read_csv(Path(render_cache_manifest_csv)):
        query_id = str(row.get("query_id", "")).strip()
        path = str(row.get("rgb_depth_cache_path", "")).strip()
        if query_id and path:
            mapping[query_id] = Path(path)
    return mapping


def build_d0_rows_from_match_table(
    *,
    match_table_csv: Path,
    output_rows_csv: Path,
    image_root: Path,
    render_cache_manifest_csv: Path | None,
    pose_by_query: Mapping[str, np.ndarray],
    camera_by_query: Mapping[str, ColmapCamera],
    stride4_rgb_cache_dir: Path | None = None,
    query_radio_dual_feature_cache_dir: Path | None = None,
    render_radio_dual_feature_cache_dir: Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build D0 probe rows with explicit GT query projections.

    The generated rows are suitable input for run_d0_probe. The function can
    also materialize a non-trained stride-4 RGB feature cache for a baseline.
    """

    rows = _read_csv(Path(match_table_csv), max_rows=max_rows)
    render_cache_by_query = _render_cache_by_query(render_cache_manifest_csv)
    output_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    query_stride4_cache_paths: dict[str, Path] = {}
    render_stride4_cache_paths: dict[str, Path] = {}
    for source_row in rows:
        query_id = str(source_row.get("query_id", "")).strip()
        if not query_id:
            skipped["missing_query_id"] = skipped.get("missing_query_id", 0) + 1
            continue
        if query_id not in pose_by_query:
            skipped["missing_pose"] = skipped.get("missing_pose", 0) + 1
            continue
        if query_id not in camera_by_query:
            skipped["missing_camera"] = skipped.get("missing_camera", 0) + 1
            continue
        point = np.asarray([_float(source_row, "world_x"), _float(source_row, "world_y"), _float(source_row, "world_z")], dtype=np.float64)
        projected, valid = project_world_points_to_image(point.reshape(1, 3), np.asarray(pose_by_query[query_id]), camera_by_query[query_id])
        if not bool(valid[0]):
            skipped["invalid_projection"] = skipped.get("invalid_projection", 0) + 1
            continue
        output_row: dict[str, Any] = {
            "query_id": query_id,
            "render_x": _float(source_row, "render_x"),
            "render_y": _float(source_row, "render_y"),
            "center_x": _float(source_row, "render_x"),
            "center_y": _float(source_row, "render_y"),
            "query_gt_x": float(projected[0, 0]),
            "query_gt_y": float(projected[0, 1]),
        }
        if stride4_rgb_cache_dir is not None:
            if query_id not in query_stride4_cache_paths:
                image_path = Path(image_root) / query_id
                if not image_path.exists():
                    skipped["missing_query_image"] = skipped.get("missing_query_image", 0) + 1
                    continue
                query_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                query_stride4_cache_paths[query_id] = _write_stride4_rgb_cache(
                    Path(stride4_rgb_cache_dir) / "query" / f"{_safe_image_stem(query_id)}_stride4_rgb.npz",
                    query_rgb,
                )
            if query_id not in render_stride4_cache_paths:
                render_cache_path = render_cache_by_query.get(query_id)
                if render_cache_path is None or not render_cache_path.exists():
                    skipped["missing_render_rgb_depth_cache"] = skipped.get("missing_render_rgb_depth_cache", 0) + 1
                    continue
                with np.load(render_cache_path) as data:
                    render_rgb = np.asarray(data["rgb"], dtype=np.uint8)
                render_stride4_cache_paths[query_id] = _write_stride4_rgb_cache(
                    Path(stride4_rgb_cache_dir) / "render" / f"{_safe_image_stem(query_id)}_stride4_rgb.npz",
                    render_rgb,
                )
            output_row["query_stride4_rgb_feature_cache_path"] = str(query_stride4_cache_paths[query_id])
            output_row["render_stride4_rgb_feature_cache_path"] = str(render_stride4_cache_paths[query_id])
        if query_radio_dual_feature_cache_dir is not None:
            stem = _safe_image_stem(query_id)
            query_radio_candidates = [
                Path(query_radio_dual_feature_cache_dir)
                / f"{stem}_{int(camera_by_query[query_id].width)}x{int(camera_by_query[query_id].height)}_radio_dual.npz",
                Path(query_radio_dual_feature_cache_dir)
                / f"{stem}_{int(camera_by_query[query_id].width)}x{int(camera_by_query[query_id].height)}.npz",
            ]
            output_row["query_radio_dual_feature_cache_path"] = str(
                next((candidate for candidate in query_radio_candidates if candidate.exists()), query_radio_candidates[0])
            )
        if render_radio_dual_feature_cache_dir is not None:
            stem = _safe_image_stem(query_id)
            output_row["render_radio_dual_feature_cache_path"] = str(
                Path(render_radio_dual_feature_cache_dir) / f"{stem}_{int(camera_by_query[query_id].width)}x{int(camera_by_query[query_id].height)}_radio_dual.npz"
            )
        output_rows.append(output_row)
    fieldnames = [
        "query_id",
        "render_x",
        "render_y",
        "center_x",
        "center_y",
        "query_gt_x",
        "query_gt_y",
        "query_stride4_rgb_feature_cache_path",
        "render_stride4_rgb_feature_cache_path",
        "query_radio_dual_feature_cache_path",
        "render_radio_dual_feature_cache_path",
    ]
    _write_csv(Path(output_rows_csv), output_rows, fieldnames)
    return {
        "input_rows": int(len(rows)),
        "output_rows": int(len(output_rows)),
        "skipped": dict(sorted(skipped.items())),
        "output_rows_csv": str(output_rows_csv),
    }
