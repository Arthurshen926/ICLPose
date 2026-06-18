from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_fixed,
    estimate_pose_pnp_fixed_robust,
    estimate_pose_pnp_ransac,
    match_reprojection_errors,
    pnp_pose_error,
)
from feature_extract.vfm.rendered_keypoint_matching import backproject_depth_to_world


def project_world_to_image(points_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> np.ndarray:
    """Project world points with a COLMAP camera and a world-to-camera pose."""

    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix, distortion = camera_matrix_and_distortion(camera)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for projection diagnostics") from exc

    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(points, rvec, pose[:3, 3], matrix, distortion)
    return projected.reshape(-1, 2).astype(np.float64)


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values.astype(np.float64), float(percentile)))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values.astype(np.float64)))


def render_depth_roundtrip_stats(
    xy: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
    pose_w2c: np.ndarray,
) -> dict[str, float | int]:
    """Backproject render pixels with depth, then project them back with the same pose."""

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    world, valid = backproject_depth_to_world(coords, depth, camera, pose_w2c)
    projected = project_world_to_image(world[valid], pose_w2c, camera) if np.any(valid) else np.zeros((0, 2))
    errors = np.linalg.norm(projected - coords[valid], axis=1) if projected.size else np.zeros((0,), dtype=np.float64)
    count = int(coords.shape[0])
    valid_count = int(np.count_nonzero(valid))
    return {
        "count": count,
        "valid_count": valid_count,
        "valid_fraction": float(valid_count / count) if count else 0.0,
        "median_roundtrip_error_px": _safe_percentile(errors, 50.0),
        "p95_roundtrip_error_px": _safe_percentile(errors, 95.0),
        "max_roundtrip_error_px": float(np.max(errors)) if errors.size else float("nan"),
    }


def _matches_from_render_pixels(
    xy: np.ndarray,
    xyz: np.ndarray,
    valid: Sequence[bool],
    *,
    source: str = "synthetic_render_depth",
    measurement_sigma_px: float | None = None,
) -> list[QueryTo3DMatch]:
    matches: list[QueryTo3DMatch] = []
    for index, is_valid in enumerate(np.asarray(valid, dtype=bool).reshape(-1)):
        if not bool(is_valid):
            continue
        matches.append(
            QueryTo3DMatch(
                token_index=index,
                xy=np.asarray(xy[index], dtype=np.float64),
                track_id=index,
                xyz=np.asarray(xyz[index], dtype=np.float64),
                similarity=1.0,
                ratio=1.0,
                landmark_variance=0.0,
                source=str(source),
                pnp_soft_score=1.0,
                measurement_sigma_px=measurement_sigma_px,
            )
        )
    return matches


def ideal_depth_correspondences(
    xy: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
    pose_w2c: np.ndarray,
    *,
    alpha: np.ndarray | None = None,
    min_alpha: float = 0.0,
    source: str = "ideal_render_depth",
    measurement_sigma_px: float | None = 1.0,
) -> tuple[list[QueryTo3DMatch], dict[str, float | int]]:
    """Build ideal query-pixel to depth-backprojected 3D correspondences.

    This intentionally bypasses feature matching. It answers a narrower question:
    whether the rendered depth, camera model, backprojection and PnP solver can
    recover the pose when 2D observations are perfect.
    """

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    depth_values = np.asarray(depth, dtype=np.float64).reshape(-1)
    if depth_values.shape[0] != coords.shape[0]:
        raise ValueError("depth must contain one value per xy coordinate")
    alpha_mask = np.ones((coords.shape[0],), dtype=bool)
    if alpha is not None:
        alpha_values = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_values.shape[0] != coords.shape[0]:
            raise ValueError("alpha must contain one value per xy coordinate")
        alpha_mask = np.isfinite(alpha_values) & (alpha_values >= float(min_alpha))
    xyz, valid = backproject_depth_to_world(coords, depth_values, camera, pose_w2c)
    valid = valid & alpha_mask
    matches = _matches_from_render_pixels(
        coords,
        xyz,
        valid,
        source=source,
        measurement_sigma_px=measurement_sigma_px,
    )
    count = int(coords.shape[0])
    valid_count = int(np.count_nonzero(valid))
    depths = depth_values[valid]
    summary: dict[str, float | int] = {
        "input_count": count,
        "valid_match_count": valid_count,
        "valid_fraction": float(valid_count / count) if count else 0.0,
        "depth_median_m": _safe_percentile(depths, 50.0),
        "depth_p05_m": _safe_percentile(depths, 5.0),
        "depth_p95_m": _safe_percentile(depths, 95.0),
    }
    return matches, summary


def _match_weight(match: QueryTo3DMatch) -> float:
    sigma = match.measurement_sigma_px
    if sigma is not None and np.isfinite(float(sigma)) and float(sigma) > 0.0:
        return 1.0 / max(float(sigma) * float(sigma), 1e-12)
    score = match.pnp_soft_score
    if score is not None and np.isfinite(float(score)):
        return max(float(score), 1e-6)
    return 1.0


def _pnp_result_row(
    name: str,
    result,
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    gt_pose_w2c: np.ndarray | None,
) -> dict[str, float | int | bool | str | None]:
    if result.pose_w2c is not None:
        residuals = match_reprojection_errors(matches, result.pose_w2c, camera)
    else:
        residuals = np.zeros((0,), dtype=np.float64)
    if gt_pose_w2c is None:
        translation_error = None
        rotation_error = None
    else:
        error = pnp_pose_error(result.pose_w2c, gt_pose_w2c)
        translation_error = float(error.translation_m)
        rotation_error = float(error.rotation_deg)
    inlier_mask = np.asarray(result.inlier_mask, dtype=bool).reshape(-1)
    if residuals.shape[0] == inlier_mask.shape[0] and np.any(inlier_mask):
        inlier_residuals = residuals[inlier_mask]
    else:
        inlier_residuals = np.zeros((0,), dtype=np.float64)
    return {
        "solver": str(name),
        "success": bool(result.success),
        "match_count": int(result.match_count),
        "inlier_count": int(result.inlier_count),
        "inlier_ratio": float(result.inlier_ratio),
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "residual_mean_px": _safe_mean(residuals),
        "residual_median_px": _safe_percentile(residuals, 50.0),
        "residual_p90_px": _safe_percentile(residuals, 90.0),
        "residual_p95_px": _safe_percentile(residuals, 95.0),
        "inlier_residual_median_px": _safe_percentile(inlier_residuals, 50.0),
    }


def _estimate_pose_pnp_magsac(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    reprojection_error_px: float,
    confidence: float,
    iterations: int,
):
    """Run OpenCV's USAC/MAGSAC PnP overload when available."""

    if len(matches) < 4:
        from feature_extract.vfm.query_to_3d_matching import PnPResult

        return PnPResult(False, None, np.zeros((len(matches),), dtype=bool), len(matches), 0)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for MAGSAC PnP") from exc
    if not hasattr(cv2, "UsacParams") or not hasattr(cv2, "USAC_MAGSAC"):
        return estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=reprojection_error_px,
            confidence=confidence,
            iterations=iterations,
            pnp_method="EPNP",
            refine_method="LM",
        )
    from feature_extract.vfm.query_to_3d_matching import PnPResult, camera_matrix_and_distortion, deduplicate_pnp_matches

    unique_matches, original_indices = deduplicate_pnp_matches(matches)
    if len(unique_matches) < 4:
        return PnPResult(False, None, np.zeros((len(matches),), dtype=bool), len(matches), 0)
    object_points = np.stack([match.xyz for match in unique_matches], axis=0).astype(np.float64)
    image_points = np.stack([match.xy for match in unique_matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    params = cv2.UsacParams()
    params.confidence = float(confidence)
    params.maxIterations = int(iterations)
    params.threshold = float(reprojection_error_px)
    params.score = int(cv2.USAC_MAGSAC)
    try:
        output = cv2.solvePnPRansac(object_points, image_points, camera_matrix, distortion, None, None, None, params)
    except Exception:
        return estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=reprojection_error_px,
            confidence=confidence,
            iterations=iterations,
            pnp_method="EPNP",
            refine_method="LM",
        )
    if len(output) == 5:
        success, _camera_matrix, rvec, tvec, inliers = output
    else:
        success, rvec, tvec, inliers = output
    if not success or rvec is None or tvec is None or inliers is None:
        return PnPResult(False, None, np.zeros((len(matches),), dtype=bool), len(matches), 0)
    rotation, _jacobian = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.astype(np.float64)
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    mask = np.zeros((len(matches),), dtype=bool)
    unique_inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    mask[original_indices[unique_inliers]] = True
    return PnPResult(True, pose, mask, len(matches), int(mask.sum()))


def run_pnp_solver_ablation(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    gt_pose_w2c: np.ndarray | None = None,
    solvers: Sequence[str] = ("plain", "ransac", "magsac", "weighted", "covariance", "oracle_uncertainty"),
    reprojection_error_px: float = 8.0,
    confidence: float = 0.999,
    iterations: int = 1000,
) -> dict[str, dict[str, float | int | bool | str | None]]:
    """Run a PnP solver matrix on a fixed correspondence set."""

    values = list(matches)
    rows: dict[str, dict[str, float | int | bool | str | None]] = {}
    gt_pose = None if gt_pose_w2c is None else np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    for solver in solvers:
        name = str(solver).lower()
        if name == "plain":
            result = estimate_pose_pnp_fixed(values, camera, min_inliers=4, pnp_method="ITERATIVE", refine_method="LM")
        elif name == "ransac":
            result = estimate_pose_pnp_ransac(
                values,
                camera,
                reprojection_error_px=float(reprojection_error_px),
                confidence=float(confidence),
                iterations=int(iterations),
                pnp_method="EPNP",
                refine_method="LM",
            )
        elif name == "magsac":
            result = _estimate_pose_pnp_magsac(
                values,
                camera,
                reprojection_error_px=float(reprojection_error_px),
                confidence=float(confidence),
                iterations=int(iterations),
            )
        elif name == "weighted":
            weights = np.asarray([_match_weight(match) for match in values], dtype=np.float64)
            result = estimate_pose_pnp_fixed_robust(
                values,
                camera,
                weights=weights,
                min_inliers=4,
                pnp_method="EPNP",
                loss="huber",
                f_scale_px=max(float(reprojection_error_px) * 0.5, 1.0),
            )
        elif name == "covariance":
            weights = np.asarray(
                [
                    1.0
                    / max(
                        float(match.measurement_sigma_px or max(float(reprojection_error_px) * 0.5, 1.0)) ** 2,
                        1e-12,
                    )
                    for match in values
                ],
                dtype=np.float64,
            )
            result = estimate_pose_pnp_fixed_robust(
                values,
                camera,
                weights=weights,
                min_inliers=4,
                pnp_method="EPNP",
                loss="linear",
                f_scale_px=max(float(reprojection_error_px), 1.0),
            )
        elif name == "oracle_uncertainty":
            if gt_pose is None:
                weights = np.ones((len(values),), dtype=np.float64)
            else:
                gt_residuals = match_reprojection_errors(values, gt_pose, camera)
                oracle_sigma = np.maximum(gt_residuals, 0.25)
                weights = 1.0 / np.maximum(oracle_sigma * oracle_sigma, 1e-12)
            result = estimate_pose_pnp_fixed_robust(
                values,
                camera,
                weights=weights,
                min_inliers=4,
                initial_pose_w2c=gt_pose,
                pnp_method="EPNP",
                loss="linear",
                f_scale_px=1.0,
            )
        else:
            raise ValueError(f"unsupported PnP ablation solver: {solver}")
        rows[name] = _pnp_result_row(name, result, values, camera, gt_pose)
    return rows


def _calibration_ece(scores: np.ndarray, labels: np.ndarray, bin_count: int) -> float:
    bins = max(int(bin_count), 1)
    total = int(scores.size)
    if total == 0:
        return float("nan")
    ece = 0.0
    for bin_idx in range(bins):
        lo = float(bin_idx) / float(bins)
        hi = float(bin_idx + 1) / float(bins)
        if bin_idx == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(scores[mask])) - float(np.mean(labels[mask])))
    return float(ece)


def match_validity_calibration_stats(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    scores: Sequence[float] | np.ndarray | None = None,
    thresholds_px: Sequence[float] = (5.0, 10.0),
    bin_count: int = 10,
) -> dict[str, float | int]:
    """Report match-level validity calibration against reprojection thresholds."""

    values = list(matches)
    if scores is None:
        score_values = np.asarray(
            [
                float(match.pnp_soft_score)
                if match.pnp_soft_score is not None and np.isfinite(float(match.pnp_soft_score))
                else float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0))
                for match in values
            ],
            dtype=np.float64,
        )
    else:
        score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_values.shape[0] != len(values):
            raise ValueError("scores must contain one value per match")
    score_values = np.clip(score_values, 0.0, 1.0)
    residuals = match_reprojection_errors(values, pose_w2c, camera) if values else np.zeros((0,), dtype=np.float64)
    stats: dict[str, float | int] = {
        "match_count": int(len(values)),
        "score_mean": _safe_mean(score_values),
        "residual_median_px": _safe_percentile(residuals, 50.0),
        "residual_p90_px": _safe_percentile(residuals, 90.0),
    }
    for threshold in thresholds_px:
        threshold_value = float(threshold)
        prefix = f"validity_{int(threshold_value) if threshold_value.is_integer() else threshold_value:g}px"
        labels = (residuals <= threshold_value).astype(np.float64)
        brier = np.mean((score_values - labels) ** 2) if labels.size else float("nan")
        stats[f"{prefix}_positive_rate"] = _safe_mean(labels)
        stats[f"{prefix}_brier"] = float(brier)
        stats[f"{prefix}_ece"] = _calibration_ece(score_values, labels, int(bin_count))
    return stats


def _threshold_tag(threshold_px: float) -> str:
    value = float(threshold_px)
    if value.is_integer():
        return f"{int(value)}px"
    return f"{value:g}px".replace(".", "p")


def oracle_fine_matches(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    gt_pose_w2c: np.ndarray,
    *,
    measurement_sigma_px: float = 0.25,
) -> list[QueryTo3DMatch]:
    """Snap match observations to the GT projection of their current 3D points."""

    values = list(matches)
    if not values:
        return []
    projected = project_world_to_image(
        np.stack([match.xyz for match in values], axis=0).astype(np.float64),
        np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4),
        camera,
    )
    snapped = []
    for match, xy in zip(values, projected):
        if not np.all(np.isfinite(xy)):
            continue
        snapped.append(
            replace(
                match,
                xy=np.asarray(xy, dtype=np.float64),
                source=f"{match.source}:oracle_fine",
                pnp_soft_score=1.0,
                measurement_sigma_px=float(measurement_sigma_px),
            )
        )
    return snapped


def run_oracle_correspondence_pnp_diagnostics(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    gt_pose_w2c: np.ndarray,
    thresholds_px: Sequence[float] = (5.0, 10.0, 16.0),
    solvers: Sequence[str] = ("ransac", "weighted", "oracle_uncertainty"),
    reprojection_error_px: float = 4.0,
    confidence: float = 0.999,
    iterations: int = 1000,
) -> dict[str, dict[str, float | int | bool | str | None]]:
    """Run upper-bound PnP diagnostics for match identity, fine offsets and solver weighting."""

    values = list(matches)
    gt_pose = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    residuals = match_reprojection_errors(values, gt_pose, camera) if values else np.zeros((0,), dtype=np.float64)
    cases: dict[str, list[QueryTo3DMatch]] = {
        "all": values,
        "oracle_fine_all": oracle_fine_matches(values, camera, gt_pose),
    }
    for threshold in thresholds_px:
        tag = _threshold_tag(float(threshold))
        keep = np.isfinite(residuals) & (residuals <= float(threshold))
        subset = [match for match, is_kept in zip(values, keep) if bool(is_kept)]
        cases[f"oracle_match_{tag}"] = subset
        cases[f"oracle_match_{tag}_oracle_fine"] = oracle_fine_matches(subset, camera, gt_pose)

    rows: dict[str, dict[str, float | int | bool | str | None]] = {}
    for case_name, case_matches in cases.items():
        ablation = run_pnp_solver_ablation(
            case_matches,
            camera,
            gt_pose_w2c=gt_pose,
            solvers=solvers,
            reprojection_error_px=float(reprojection_error_px),
            confidence=float(confidence),
            iterations=int(iterations),
        )
        for solver_name, row in ablation.items():
            key = f"{case_name}_{solver_name}"
            rows[key] = {**row, "case": case_name}
    return rows


def synthetic_render_lock_stats(
    xy: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    min_inliers: int = 4,
    pnp_method: str = "ITERATIVE",
    refine_method: str = "lm",
) -> dict[str, float | int | bool]:
    """Solve PnP from query pixels paired to 3D points backprojected from the same render pose.

    This isolates a structural failure mode: if correspondence pixels are not corrected away from
    the render grid, depth-derived 3D points make a geometrically self-consistent PnP problem whose
    optimum is the render pose.
    """

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xyz, valid = backproject_depth_to_world(coords, depth, camera, render_pose_w2c)
    matches = _matches_from_render_pixels(coords, xyz, valid)
    result = estimate_pose_pnp_fixed(
        matches,
        camera,
        min_inliers=int(min_inliers),
        pnp_method=pnp_method,
        refine_method=refine_method,
    )
    error = pnp_pose_error(result.pose_w2c, np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4))
    return {
        "match_count": len(matches),
        "pnp_success": bool(result.success),
        "pnp_inlier_count": int(result.inlier_count),
        "pnp_render_translation_delta_m": float(error.translation_m),
        "pnp_render_rotation_delta_deg": float(error.rotation_deg),
    }
