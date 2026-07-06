from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation, qvec_to_rotmat
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, estimate_pose_pnp_ransac, pnp_pose_error


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def deduplicate_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    dustbin_threshold: float = 0.5,
) -> list[dict[str, object]]:
    best: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        track_text = str(row.get("track_id", "")).strip()
        if not query_id or not track_text:
            continue
        dustbin = _float(row, "dustbin_probability", default=1.0)
        if dustbin >= float(dustbin_threshold):
            continue
        key = (query_id, int(track_text))
        existing = best.get(key)
        if existing is None or dustbin < _float(existing, "dustbin_probability", default=1.0):
            best[key] = dict(row)
    return [best[key] for key in sorted(best)]


def scaled_colmap_camera(camera: ColmapCamera, *, image_width: int, image_height: int) -> ColmapCamera:
    sx = float(image_width) / float(camera.width)
    sy = float(image_height) / float(camera.height)
    params = list(float(value) for value in camera.params)
    if int(camera.model_id) in {0, 2, 3, 7, 8, 9} and len(params) >= 3:
        scale = 0.5 * (sx + sy)
        params[0] *= scale
        params[1] *= sx
        params[2] *= sy
    elif len(params) >= 4:
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    return ColmapCamera(
        camera_id=int(camera.camera_id),
        model_id=int(camera.model_id),
        width=int(image_width),
        height=int(image_height),
        params=tuple(params),
    )


def pose_w2c_from_colmap_image(image: ColmapImageObservation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(np.asarray(image.qvec, dtype=np.float64))
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    return pose


def _matches_for_query(
    rows: Sequence[Mapping[str, object]],
    *,
    xyz_by_track: Mapping[int, np.ndarray],
) -> list[QueryTo3DMatch]:
    matches: list[QueryTo3DMatch] = []
    for idx, row in enumerate(rows):
        track_id = int(str(row.get("track_id", "")).strip())
        xyz = xyz_by_track.get(track_id)
        if xyz is None:
            continue
        dustbin = _float(row, "dustbin_probability", default=1.0)
        matches.append(
            QueryTo3DMatch(
                token_index=int(idx),
                xy=np.asarray([_float(row, "query_pred_x"), _float(row, "query_pred_y")], dtype=np.float64),
                track_id=track_id,
                xyz=np.asarray(xyz, dtype=np.float64).reshape(3),
                similarity=float(1.0 - dustbin),
                ratio=1.0,
                landmark_variance=0.0,
                source="measurement_v1_rgb_patch_proxy",
                pnp_soft_score=float(1.0 - dustbin),
            )
        )
    return matches


def evaluate_pose_proxy_from_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    cameras: Mapping[int, ColmapCamera],
    images_by_name: Mapping[str, ColmapImageObservation],
    xyz_by_track: Mapping[int, np.ndarray],
    image_width: int,
    image_height: int,
    dustbin_threshold: float = 0.5,
    reprojection_error_px: float = 8.0,
    min_inliers: int = 4,
) -> dict[str, Any]:
    kept = deduplicate_prediction_rows(rows, dustbin_threshold=float(dustbin_threshold))
    by_query: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in kept:
        by_query[str(row.get("query_id", "")).strip()].append(row)
    pose_rows: list[dict[str, Any]] = []
    matched_measurement_count = 0
    missing_xyz_count = 0
    missing_query_count = 0
    missing_camera_count = 0
    for query_id in sorted(by_query):
        image = images_by_name.get(query_id)
        if image is None:
            missing_query_count += 1
            pose_rows.append(
                {
                    "query_id": query_id,
                    "match_count": 0,
                    "success": False,
                    "inlier_count": 0,
                    "translation_error_m": float("inf"),
                    "rotation_error_deg": float("inf"),
                    "failure_reason": "missing_query_pose",
                }
            )
            continue
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            missing_camera_count += 1
            pose_rows.append(
                {
                    "query_id": query_id,
                    "match_count": 0,
                    "success": False,
                    "inlier_count": 0,
                    "translation_error_m": float("inf"),
                    "rotation_error_deg": float("inf"),
                    "failure_reason": "missing_camera",
                }
            )
            continue
        scaled_camera = scaled_colmap_camera(camera, image_width=int(image_width), image_height=int(image_height))
        matches = _matches_for_query(by_query[query_id], xyz_by_track=xyz_by_track)
        matched_measurement_count += int(len(matches))
        missing_xyz_count += int(len(by_query[query_id]) - len(matches))
        pnp = estimate_pose_pnp_ransac(
            matches,
            scaled_camera,
            reprojection_error_px=float(reprojection_error_px),
            iterations=1000,
            min_inliers=int(min_inliers),
            refine_lm=True,
        )
        error = pnp_pose_error(pnp.pose_w2c if pnp.success else None, pose_w2c_from_colmap_image(image))
        pose_rows.append(
            {
                "query_id": query_id,
                "match_count": int(len(matches)),
                "success": bool(pnp.success),
                "inlier_count": int(pnp.inlier_count),
                "translation_error_m": float(error.translation_m),
                "rotation_error_deg": float(error.rotation_deg),
                "failure_reason": "" if bool(pnp.success) else "pnp_failed",
            }
        )
    successful = [row for row in pose_rows if bool(row["success"])]
    translations = np.asarray([float(row["translation_error_m"]) for row in successful], dtype=np.float64)
    rotations = np.asarray([float(row["rotation_error_deg"]) for row in successful], dtype=np.float64)
    match_counts = np.asarray([int(row["match_count"]) for row in pose_rows], dtype=np.float64)

    def _rate(max_translation_m: float, max_rotation_deg: float) -> float:
        if not pose_rows:
            return 0.0
        passed = 0
        for row in pose_rows:
            if not bool(row["success"]):
                continue
            if float(row["translation_error_m"]) <= float(max_translation_m) and float(row["rotation_error_deg"]) <= float(max_rotation_deg):
                passed += 1
        return float(passed / len(pose_rows))

    return {
        "query_count": int(len(pose_rows)),
        "success_count": int(len(successful)),
        "success_rate": float(len(successful) / len(pose_rows)) if pose_rows else 0.0,
        "median_translation_error_m": float(np.median(translations)) if translations.size else float("inf"),
        "median_rotation_error_deg": float(np.median(rotations)) if rotations.size else float("inf"),
        "translation_error_p90_m": float(np.percentile(translations, 90.0)) if translations.size else float("inf"),
        "rotation_error_p90_deg": float(np.percentile(rotations, 90.0)) if rotations.size else float("inf"),
        "rate_3cm_1deg": _rate(0.03, 1.0),
        "rate_5cm_2deg": _rate(0.05, 2.0),
        "rate_10cm_5deg": _rate(0.10, 5.0),
        "median_match_count": float(np.median(match_counts)) if match_counts.size else 0.0,
        "match_count_p10": float(np.percentile(match_counts, 10.0)) if match_counts.size else 0.0,
        "pose_rows": pose_rows,
        "kept_measurement_count": int(len(kept)),
        "matched_measurement_count": int(matched_measurement_count),
        "missing_xyz_count": int(missing_xyz_count),
        "missing_query_count": int(missing_query_count),
        "missing_camera_count": int(missing_camera_count),
        "dustbin_threshold": float(dustbin_threshold),
    }
