from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


LOCAL_AFFINE_FIELDNAMES = [
    "support_to_query_a00",
    "support_to_query_a01",
    "support_to_query_a10",
    "support_to_query_a11",
    "local_affine_points",
    "local_affine_rmse_px",
    "local_affine_det",
    "local_affine_valid",
]

LOCAL_HOMOGRAPHY_FIELDNAMES = [
    "support_to_query_h00",
    "support_to_query_h01",
    "support_to_query_h02",
    "support_to_query_h10",
    "support_to_query_h11",
    "support_to_query_h12",
    "support_to_query_h20",
    "support_to_query_h21",
    "support_to_query_h22",
    "local_homography_points",
    "local_homography_rmse_px",
    "local_homography_det",
    "local_homography_valid",
]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _float(row: Mapping[str, object], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def _support_xy(row: Mapping[str, object]) -> tuple[float, float]:
    support_x = str(row.get("support_x", "")).strip()
    support_y = str(row.get("support_y", "")).strip()
    if support_x and support_y:
        return float(support_x), float(support_y)
    return _float(row, "render_x"), _float(row, "render_y")


def _target_pair(row: Mapping[str, object]) -> tuple[str, str]:
    support_id = str(row.get("support_image_id", "")).strip()
    query_id = str(row.get("query_id", "")).strip()
    if not support_id or not query_id:
        raise ValueError("rows must contain support_image_id and query_id")
    return support_id, query_id


def _scaled_xy(obs: ColmapTrackObservation, *, image_width: int | None, image_height: int | None) -> tuple[float, float]:
    target_w = int(image_width if image_width is not None else (obs.image_width or 0))
    target_h = int(image_height if image_height is not None else (obs.image_height or 0))
    source_w = int(obs.image_width or target_w)
    source_h = int(obs.image_height or target_h)
    if target_w <= 0 or target_h <= 0 or source_w <= 0 or source_h <= 0:
        raise ValueError("image dimensions must be provided either as arguments or observation metadata")
    return float(obs.xy[0]) * float(target_w) / float(source_w), float(obs.xy[1]) * float(target_h) / float(source_h)


def _pair_points_for_rows(
    observations: Sequence[ColmapTrackObservation],
    pair_keys: set[tuple[str, str]],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[tuple[str, str], list[tuple[float, float, float, float]]]:
    partners_by_support: dict[str, set[str]] = defaultdict(set)
    for support_id, query_id in pair_keys:
        partners_by_support[str(support_id)].add(str(query_id))

    grouped: dict[int, list[ColmapTrackObservation]] = defaultdict(list)
    for obs in observations:
        grouped[int(obs.track_id)].append(obs)

    pair_points: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for track_obs in grouped.values():
        by_image: dict[str, ColmapTrackObservation] = {}
        for obs in track_obs:
            by_image.setdefault(str(obs.image_id), obs)
        for support_id, query_ids in partners_by_support.items():
            support = by_image.get(support_id)
            if support is None:
                continue
            sx, sy = _scaled_xy(support, image_width=image_width, image_height=image_height)
            for query_id in query_ids:
                query = by_image.get(query_id)
                if query is None:
                    continue
                qx, qy = _scaled_xy(query, image_width=image_width, image_height=image_height)
                pair_points[(support_id, query_id)].append((sx, sy, qx, qy))
    return pair_points


def _pair_points_from_colmap_model(
    model_dir: Path,
    pair_keys: set[tuple[str, str]],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[tuple[str, str], list[tuple[float, float, float, float]]]:
    """Load only image-level intersections needed by a diagnostic pair set."""

    images = read_colmap_images_binary(Path(model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(model_dir) / "cameras.bin")
    required_images = {image_id for pair in pair_keys for image_id in pair}
    by_name = {str(image.image_name): image for image in images.values()}
    track_xy_by_image: dict[str, dict[int, tuple[float, float]]] = {}
    for image_id in required_images:
        image = by_name.get(str(image_id))
        if image is None:
            continue
        camera = cameras.get(int(image.camera_id))
        source_width = int(camera.width) if camera is not None else int(image_width or 0)
        source_height = int(camera.height) if camera is not None else int(image_height or 0)
        target_width = int(image_width or source_width)
        target_height = int(image_height or source_height)
        if min(source_width, source_height, target_width, target_height) <= 0:
            raise ValueError("COLMAP affine oracle requires valid image dimensions")
        scale_x = float(target_width) / float(source_width)
        scale_y = float(target_height) / float(source_height)
        track_xy_by_image[str(image_id)] = {
            int(track_id): (float(xy[0]) * scale_x, float(xy[1]) * scale_y)
            for track_id, xy in zip(image.point3d_ids.tolist(), image.xys.tolist())
            if int(track_id) >= 0
        }
    pair_points: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    for support_id, query_id in pair_keys:
        support = track_xy_by_image.get(str(support_id), {})
        query = track_xy_by_image.get(str(query_id), {})
        common = sorted(set(support) & set(query))
        pair_points[(str(support_id), str(query_id))] = [
            (*support[track_id], *query[track_id]) for track_id in common
        ]
    return pair_points


def _fit_local_support_to_query_affine(
    points: Sequence[tuple[float, float, float, float]],
    *,
    support_anchor_xy: tuple[float, float],
    query_anchor_xy: tuple[float, float],
    local_radius_px: float,
    min_points: int,
    max_points: int,
) -> dict[str, Any] | None:
    if len(points) < int(min_points):
        return None
    values = np.asarray(points, dtype=np.float64)
    support_anchor = np.asarray(support_anchor_xy, dtype=np.float64).reshape(1, 2)
    query_anchor = np.asarray(query_anchor_xy, dtype=np.float64).reshape(1, 2)
    support_offsets = values[:, 0:2] - support_anchor
    query_offsets = values[:, 2:4] - query_anchor
    distance = np.linalg.norm(support_offsets, axis=1)
    query_distance = np.linalg.norm(query_offsets, axis=1)
    local = np.maximum(distance, query_distance) <= float(local_radius_px)
    if int(np.sum(local)) >= int(min_points):
        selected = np.flatnonzero(local)
        selected = selected[np.argsort(distance[selected])[: int(max_points)]]
    else:
        selected = np.argsort(distance)[: int(max_points)]
    if int(selected.size) < int(min_points):
        return None
    x = support_offsets[selected]
    y = query_offsets[selected]
    if np.linalg.matrix_rank(x) < 2:
        return None
    sigma = max(float(local_radius_px), 1e-6) * 0.5
    weights = np.exp(-0.5 * (distance[selected] / sigma) ** 2)
    sqrt_w = np.sqrt(np.maximum(weights, 1e-6)).reshape(-1, 1)
    solution, *_ = np.linalg.lstsq(x * sqrt_w, y * sqrt_w, rcond=None)
    matrix = solution.T
    predicted = x @ solution
    residual = predicted - y
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    det = float(np.linalg.det(matrix))
    if not np.isfinite(matrix).all() or not np.isfinite(rmse) or abs(det) < 1e-6:
        return None
    return {
        "support_to_query_a00": float(matrix[0, 0]),
        "support_to_query_a01": float(matrix[0, 1]),
        "support_to_query_a10": float(matrix[1, 0]),
        "support_to_query_a11": float(matrix[1, 1]),
        "local_affine_points": int(selected.size),
        "local_affine_rmse_px": rmse,
        "local_affine_det": det,
        "local_affine_valid": True,
    }


def _normalise_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid.reshape(1, 2)
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = math.sqrt(2.0) / max(mean_distance, 1e-8)
    transform = np.asarray(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([points, np.ones((int(points.shape[0]), 1), dtype=np.float64)], axis=1)
    normalised = (transform @ homogeneous.T).T
    return normalised[:, :2], transform


def _fit_homography(support_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray | None:
    if int(support_xy.shape[0]) < 4 or int(query_xy.shape[0]) < 4:
        return None
    source_norm, source_t = _normalise_points(support_xy)
    target_norm, target_t = _normalise_points(query_xy)
    rows = []
    for (x, y), (u, v) in zip(source_norm, target_norm):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        return None
    _, _, vh = np.linalg.svd(design)
    h_norm = vh[-1].reshape(3, 3)
    try:
        h = np.linalg.inv(target_t) @ h_norm @ source_t
    except np.linalg.LinAlgError:
        return None
    if abs(float(h[2, 2])) < 1e-12:
        return None
    h = h / h[2, 2]
    if not np.isfinite(h).all():
        return None
    return h


def _homography_reprojection_rmse(homography: np.ndarray, support_xy: np.ndarray, query_xy: np.ndarray) -> float:
    homogeneous = np.concatenate([support_xy, np.ones((int(support_xy.shape[0]), 1), dtype=np.float64)], axis=1)
    projected = (homography @ homogeneous.T).T
    projected_xy = projected[:, :2] / np.clip(projected[:, 2:3], 1e-12, None)
    residual = projected_xy - query_xy
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _homography_point_error(homography: Mapping[str, Any], support_xy: tuple[float, float], query_xy: tuple[float, float]) -> float:
    h = np.asarray(
        [
            [float(homography["support_to_query_h00"]), float(homography["support_to_query_h01"]), float(homography["support_to_query_h02"])],
            [float(homography["support_to_query_h10"]), float(homography["support_to_query_h11"]), float(homography["support_to_query_h12"])],
            [float(homography["support_to_query_h20"]), float(homography["support_to_query_h21"]), float(homography["support_to_query_h22"])],
        ],
        dtype=np.float64,
    )
    support = np.asarray([float(support_xy[0]), float(support_xy[1]), 1.0], dtype=np.float64)
    projected = h @ support
    if abs(float(projected[2])) < 1e-12:
        return float("inf")
    projected_xy = projected[:2] / projected[2]
    target = np.asarray([float(query_xy[0]), float(query_xy[1])], dtype=np.float64)
    return float(np.linalg.norm(projected_xy - target))


def _fit_origin_fixed_homography(support_offsets: np.ndarray, query_offsets: np.ndarray) -> np.ndarray | None:
    if int(support_offsets.shape[0]) < 4 or int(query_offsets.shape[0]) < 4:
        return None
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(support_offsets, query_offsets):
        rows.append([x, y, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, x, y, -v * x, -v * y])
        rhs.append(v)
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 6:
        return None
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b, c, d, p, q = [float(v) for v in solution]
    h = np.asarray([[a, b, 0.0], [c, d, 0.0], [p, q, 1.0]], dtype=np.float64)
    if not np.isfinite(h).all():
        return None
    denominators = support_offsets @ h[2, :2] + 1.0
    if float(np.min(np.abs(denominators))) < 1e-8:
        return None
    return h


def _fit_local_support_to_query_homography(
    points: Sequence[tuple[float, float, float, float]],
    *,
    support_anchor_xy: tuple[float, float],
    query_anchor_xy: tuple[float, float],
    local_radius_px: float,
    min_points: int,
    max_points: int,
) -> dict[str, Any] | None:
    if len(points) < int(min_points):
        return None
    values = np.asarray(points, dtype=np.float64)
    support_anchor = np.asarray(support_anchor_xy, dtype=np.float64).reshape(1, 2)
    query_anchor = np.asarray(query_anchor_xy, dtype=np.float64).reshape(1, 2)
    support_xy = values[:, 0:2]
    query_xy = values[:, 2:4]
    support_offsets = support_xy - support_anchor
    query_offsets = query_xy - query_anchor
    distance = np.maximum(np.linalg.norm(support_offsets, axis=1), np.linalg.norm(query_offsets, axis=1))
    local = distance <= float(local_radius_px)
    if int(np.sum(local)) >= int(min_points):
        selected = np.flatnonzero(local)
        selected = selected[np.argsort(distance[selected])[: int(max_points)]]
    else:
        selected = np.argsort(distance)[: int(max_points)]
    if int(selected.size) < int(min_points):
        return None
    h_local = _fit_origin_fixed_homography(support_offsets[selected], query_offsets[selected])
    if h_local is None:
        return None
    support_anchor_h = np.asarray(
        [[1.0, 0.0, -float(support_anchor[0, 0])], [0.0, 1.0, -float(support_anchor[0, 1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    query_anchor_h = np.asarray(
        [[1.0, 0.0, float(query_anchor[0, 0])], [0.0, 1.0, float(query_anchor[0, 1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h = query_anchor_h @ h_local @ support_anchor_h
    if abs(float(h[2, 2])) < 1e-12:
        return None
    h = h / h[2, 2]
    if not np.isfinite(h).all():
        return None
    rmse = _homography_reprojection_rmse(h, support_xy[selected], query_xy[selected])
    det = float(np.linalg.det(h[:2, :2]))
    if not np.isfinite(rmse) or not np.isfinite(det) or abs(det) < 1e-8:
        return None
    return {
        "support_to_query_h00": float(h[0, 0]),
        "support_to_query_h01": float(h[0, 1]),
        "support_to_query_h02": float(h[0, 2]),
        "support_to_query_h10": float(h[1, 0]),
        "support_to_query_h11": float(h[1, 1]),
        "support_to_query_h12": float(h[1, 2]),
        "support_to_query_h20": float(h[2, 0]),
        "support_to_query_h21": float(h[2, 1]),
        "support_to_query_h22": float(h[2, 2]),
        "local_homography_points": int(selected.size),
        "local_homography_rmse_px": rmse,
        "local_homography_det": det,
        "local_homography_valid": True,
    }


def _augment_affine_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    pair_points: Mapping[tuple[str, str], Sequence[tuple[float, float, float, float]]],
    output_rows_csv: Path,
    local_radius_px: float,
    min_points: int,
    max_points: int,
    max_rmse_px: float | None,
) -> dict[str, Any]:
    out_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for row in rows:
        support_id, query_id = _target_pair(row)
        affine = _fit_local_support_to_query_affine(
            pair_points.get((support_id, query_id), []),
            support_anchor_xy=_support_xy(row),
            query_anchor_xy=(_float(row, "query_gt_x"), _float(row, "query_gt_y")),
            local_radius_px=float(local_radius_px),
            min_points=int(min_points),
            max_points=int(max_points),
        )
        if affine is None:
            skipped["affine_fit_failed"] = skipped.get("affine_fit_failed", 0) + 1
            continue
        if max_rmse_px is not None and float(affine["local_affine_rmse_px"]) > float(max_rmse_px):
            skipped["high_affine_rmse"] = skipped.get("high_affine_rmse", 0) + 1
            continue
        merged = dict(row)
        merged.update(affine)
        out_rows.append(merged)
    output_fields = list(fieldnames)
    for name in LOCAL_AFFINE_FIELDNAMES:
        if name not in output_fields:
            output_fields.append(name)
    _write_csv(Path(output_rows_csv), out_rows, output_fields)
    pair_keys = {_target_pair(row) for row in rows}
    return {
        "input_rows": int(len(rows)),
        "output_rows": int(len(out_rows)),
        "output_rows_csv": str(output_rows_csv),
        "pair_count": int(len(pair_keys)),
        "pairs_with_points": int(sum(1 for key in pair_keys if pair_points.get(key))),
        "local_radius_px": float(local_radius_px),
        "min_points": int(min_points),
        "max_points": int(max_points),
        "max_rmse_px": None if max_rmse_px is None else float(max_rmse_px),
        "skipped": dict(sorted(skipped.items())),
    }


def augment_measurement_rows_with_local_affine_from_observations(
    *,
    rows_csv: Path,
    observations: Sequence[ColmapTrackObservation],
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    local_radius_px: float = 32.0,
    min_points: int = 6,
    max_points: int = 64,
    max_rmse_px: float | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows, fieldnames = _read_csv(Path(rows_csv))
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    pair_keys = {_target_pair(row) for row in rows}
    pair_points = _pair_points_for_rows(
        observations,
        pair_keys,
        image_width=image_width,
        image_height=image_height,
    )
    return _augment_affine_rows(
        rows=rows,
        fieldnames=fieldnames,
        pair_points=pair_points,
        output_rows_csv=Path(output_rows_csv),
        local_radius_px=float(local_radius_px),
        min_points=int(min_points),
        max_points=int(max_points),
        max_rmse_px=max_rmse_px,
    )


def augment_measurement_rows_with_local_affine_from_colmap_model(
    *,
    rows_csv: Path,
    model_dir: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    local_radius_px: float = 32.0,
    min_points: int = 6,
    max_points: int = 64,
    max_rmse_px: float | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Fit a GT-observation affine upper bound; never use this in online inference."""

    rows, fieldnames = _read_csv(Path(rows_csv))
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    pair_keys = {_target_pair(row) for row in rows}
    pair_points = _pair_points_from_colmap_model(
        Path(model_dir),
        pair_keys,
        image_width=image_width,
        image_height=image_height,
    )
    summary = _augment_affine_rows(
        rows=rows,
        fieldnames=fieldnames,
        pair_points=pair_points,
        output_rows_csv=Path(output_rows_csv),
        local_radius_px=float(local_radius_px),
        min_points=int(min_points),
        max_points=int(max_points),
        max_rmse_px=max_rmse_px,
    )
    summary.update(
        {
            "diagnostic_only": True,
            "online_available": False,
            "geometry_source": "gt_colmap_query_observation_intersections",
            "model_dir": str(model_dir),
        }
    )
    return summary


def augment_measurement_rows_with_local_homography_from_observations(
    *,
    rows_csv: Path,
    observations: Sequence[ColmapTrackObservation],
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    local_radius_px: float = 32.0,
    min_points: int = 8,
    max_points: int = 64,
    max_rmse_px: float | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows, fieldnames = _read_csv(Path(rows_csv))
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    pair_keys = {_target_pair(row) for row in rows}
    pair_points = _pair_points_for_rows(observations, pair_keys, image_width=image_width, image_height=image_height)
    out_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    anchor_errors: list[float] = []
    for row in rows:
        support_id, query_id = _target_pair(row)
        support_anchor_xy = _support_xy(row)
        query_anchor_xy = (_float(row, "query_gt_x"), _float(row, "query_gt_y"))
        homography = _fit_local_support_to_query_homography(
            pair_points.get((support_id, query_id), []),
            support_anchor_xy=support_anchor_xy,
            query_anchor_xy=query_anchor_xy,
            local_radius_px=float(local_radius_px),
            min_points=int(min_points),
            max_points=int(max_points),
        )
        if homography is None:
            skipped["homography_fit_failed"] = skipped.get("homography_fit_failed", 0) + 1
            continue
        if max_rmse_px is not None and float(homography["local_homography_rmse_px"]) > float(max_rmse_px):
            skipped["high_homography_rmse"] = skipped.get("high_homography_rmse", 0) + 1
            continue
        anchor_errors.append(_homography_point_error(homography, support_anchor_xy, query_anchor_xy))
        merged = dict(row)
        merged.update(homography)
        out_rows.append(merged)
    output_fields = list(fieldnames)
    for name in LOCAL_HOMOGRAPHY_FIELDNAMES:
        if name not in output_fields:
            output_fields.append(name)
    _write_csv(Path(output_rows_csv), out_rows, output_fields)
    anchor_error_array = np.asarray(anchor_errors, dtype=np.float64)
    return {
        "input_rows": int(len(rows)),
        "output_rows": int(len(out_rows)),
        "output_rows_csv": str(output_rows_csv),
        "pair_count": int(len(pair_keys)),
        "pairs_with_points": int(sum(1 for key in pair_keys if pair_points.get(key))),
        "local_radius_px": float(local_radius_px),
        "min_points": int(min_points),
        "max_points": int(max_points),
        "max_rmse_px": None if max_rmse_px is None else float(max_rmse_px),
        "anchor_reprojection_median_px": None if int(anchor_error_array.size) == 0 else float(np.median(anchor_error_array)),
        "anchor_reprojection_max_px": None if int(anchor_error_array.size) == 0 else float(np.max(anchor_error_array)),
        "skipped": dict(sorted(skipped.items())),
    }


def augment_measurement_rows_with_local_affine_from_jsonl(
    *,
    rows_csv: Path,
    track_observations_jsonl: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    local_radius_px: float = 32.0,
    min_points: int = 6,
    max_points: int = 64,
    max_rmse_px: float | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations_jsonl(Path(track_observations_jsonl))
    return augment_measurement_rows_with_local_affine_from_observations(
        rows_csv=Path(rows_csv),
        observations=observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        local_radius_px=float(local_radius_px),
        min_points=int(min_points),
        max_points=int(max_points),
        max_rmse_px=max_rmse_px,
        max_rows=max_rows,
    )


def augment_measurement_rows_with_local_homography_from_jsonl(
    *,
    rows_csv: Path,
    track_observations_jsonl: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    local_radius_px: float = 32.0,
    min_points: int = 8,
    max_points: int = 64,
    max_rmse_px: float | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations_jsonl(Path(track_observations_jsonl))
    return augment_measurement_rows_with_local_homography_from_observations(
        rows_csv=Path(rows_csv),
        observations=observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        local_radius_px=float(local_radius_px),
        min_points=int(min_points),
        max_points=int(max_points),
        max_rmse_px=max_rmse_px,
        max_rows=max_rows,
    )


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
