from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.types import PoseEstimate, QueryMeasurement, SurfaceAnchor
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


def _matched_arrays(
    anchors: Sequence[SurfaceAnchor],
    measurements: Sequence[QueryMeasurement],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_anchor = {int(anchor.anchor_id): anchor for anchor in anchors}
    object_points = []
    image_points = []
    covariances = []
    valid_probs = []
    for measurement in measurements:
        anchor = by_anchor.get(int(measurement.anchor_id))
        if anchor is None:
            continue
        object_points.append(np.asarray(anchor.world_xyz, dtype=np.float64).reshape(3))
        image_points.append(np.asarray(measurement.query_xy_mean_px, dtype=np.float64).reshape(2))
        covariances.append(np.asarray(measurement.cov_query_2x2, dtype=np.float64).reshape(2, 2))
        valid_probs.append(float(measurement.p_valid))
    if len(object_points) < 4:
        raise ValueError("at least four anchor measurements are required for PnP")
    return (
        np.stack(object_points, axis=0).astype(np.float64),
        np.stack(image_points, axis=0).astype(np.float64),
        np.stack(covariances, axis=0).astype(np.float64),
        np.asarray(valid_probs, dtype=np.float64),
    )


def _initial_pose(object_points: np.ndarray, image_points: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for measurement_v1 PnP") from exc
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=getattr(cv2, "SOLVEPNP_EPNP"),
    )
    if not success or rvec is None or tvec is None:
        raise RuntimeError("OpenCV solvePnP failed")
    return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


def estimate_pose_from_measurements(
    anchors: Sequence[SurfaceAnchor],
    measurements: Sequence[QueryMeasurement],
    camera: ColmapCamera,
    *,
    use_covariance: bool = True,
    initial_pose_w2c: np.ndarray | None = None,
    robust_loss: str = "linear",
    max_nfev: int = 100,
) -> PoseEstimate:
    """Estimate pose from fixed anchors and query-side measurement covariances."""

    object_points, image_points, covariances, valid_probs = _matched_arrays(anchors, measurements)
    try:
        import cv2
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and SciPy are required for measurement_v1 PnP") from exc
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    if initial_pose_w2c is None:
        rvec0, tvec0 = _initial_pose(object_points, image_points, camera)
    else:
        pose0 = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
        rvec0, _jac = cv2.Rodrigues(pose0[:3, :3])
        tvec0 = pose0[:3, 3]
        rvec0 = np.asarray(rvec0, dtype=np.float64).reshape(3)
        tvec0 = np.asarray(tvec0, dtype=np.float64).reshape(3)
    params0 = np.concatenate([rvec0, tvec0]).astype(np.float64)

    if use_covariance:
        whiteners = []
        for cov, prob in zip(covariances, valid_probs):
            safe_cov = np.asarray(cov, dtype=np.float64).reshape(2, 2)
            safe_cov = safe_cov + np.eye(2, dtype=np.float64) * 1e-6
            chol = np.linalg.cholesky(safe_cov)
            whitening = np.linalg.inv(chol)
            whiteners.append(whitening * np.sqrt(max(float(prob), 1e-6)))
        whitening_stack = np.stack(whiteners, axis=0)
    else:
        whitening_stack = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], image_points.shape[0], axis=0)

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = np.asarray(params[:3], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(params[3:6], dtype=np.float64).reshape(3, 1)
        projected, _jac = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        errors = projected.reshape(-1, 2) - image_points
        whitened = np.einsum("nij,nj->ni", whitening_stack, errors)
        return whitened.reshape(-1)

    try:
        result = least_squares(
            residuals,
            params0,
            loss=str(robust_loss),
            max_nfev=int(max_nfev),
            method="trf",
        )
    except Exception:
        return PoseEstimate(
            pose_w2c=None,
            pose_cov_6x6=None,
            success=False,
            objective=float("inf"),
            inlier_probability=np.zeros((len(measurements),), dtype=np.float64),
            hessian_condition=float("inf"),
            diagnostics={"match_count": float(len(measurements))},
        )
    rvec = np.asarray(result.x[:3], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(result.x[3:6], dtype=np.float64).reshape(3, 1)
    rotation, _jac = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.astype(np.float64)
    pose[:3, 3] = tvec.reshape(3)
    jacobian = np.asarray(result.jac, dtype=np.float64)
    hessian = jacobian.T @ jacobian
    try:
        pose_cov = np.linalg.pinv(hessian)
        condition = float(np.linalg.cond(hessian))
    except Exception:
        pose_cov = None
        condition = float("inf")
    objective = float(np.mean(residuals(result.x) ** 2))
    return PoseEstimate(
        pose_w2c=pose,
        pose_cov_6x6=pose_cov,
        success=bool(result.success),
        objective=objective,
        inlier_probability=np.clip(valid_probs, 0.0, 1.0),
        hessian_condition=condition,
        diagnostics={"match_count": float(len(object_points)), "cost": float(result.cost)},
    )
