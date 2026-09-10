"""Frozen mapping-only joint image/UV covariance calibration; no query inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize


def covariance(query_variance, projected_covariance, rho=0., scale=1.):
    """PSD scalar-correlated coupling of image and projected UV errors."""
    q = np.asarray(query_variance, float).reshape(-1)
    p = np.asarray(projected_covariance, float)
    if not (-.99 <= rho <= .99) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid correlation or scale")
    if p.shape != (len(q), 2, 2) or not np.isfinite(p).all() or np.any(q <= 0):
        raise ValueError("invalid marginal covariance")
    eig, vec = np.linalg.eigh(p)
    if np.any(eig < -1e-9) or not np.isfinite(q).all():
        raise ValueError("invalid marginal covariance")
    root = (vec * np.sqrt(np.maximum(eig, 0))[:, None, :]) @ vec.transpose(0, 2, 1)
    return scale * (q[:, None, None] * np.eye(2) + p - 2 * rho * np.sqrt(q)[:, None, None] * root)


def metrics(residual, cov):
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        raise ValueError("non-SPD covariance")
    mahal = np.einsum("ni,ni->n", residual, np.linalg.solve(cov, residual[..., None])[..., 0])
    return {"nll": float(np.mean(.5 * (mahal + logdet + 2*np.log(2*np.pi)))),
            "coverage_50": float(np.mean(mahal <= 1.38629436112)),
            "coverage_90": float(np.mean(mahal <= 4.60517018599)),
            "coverage_95": float(np.mean(mahal <= 5.99146454711)),
            "mean_mahalanobis_squared": float(np.mean(mahal))}


def export_bank(path, prediction, image_scale, variance_scale, uv_scale, uv_variance_scale,
                anchor, target_world, plane_ids, planes, token_centers, observations,
                source_views, poses, matrices, radial, calibration_rows, evaluation_rows, head_hash):
    from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences import _clip_chart_coordinate_to_original_cell
    tangent = planes.frames_world[plane_ids, :2]
    anchor_uv = np.einsum("ni,nji->nj", anchor-planes.centers_world[plane_ids], tangent)
    uv = _clip_chart_coordinate_to_original_cell(anchor_uv, uv_scale*prediction[2], .5)
    predicted_world = anchor + np.einsum("ni,nij->nj", uv-anchor_uv, tangent)
    target_uv_offset = np.einsum("ni,nji->nj", target_world-anchor, tangent)
    oracle_world = anchor + np.einsum("ni,nij->nj", target_uv_offset, tangent)
    normal_residual_m = np.linalg.norm(target_world-oracle_world, axis=1)
    pixels = token_centers + image_scale*prediction[0]
    residual = np.empty_like(pixels, dtype=float)
    image_error = np.empty_like(residual)
    world_error = np.empty_like(residual)
    projected_cov = np.empty((len(anchor), 2, 2))
    pose_jacobian = np.empty((len(anchor), 2, 6))
    camera_points = np.empty((len(anchor), 3))
    offplane_image_error = np.empty_like(residual)
    for obs in np.unique(observations):
        rows = np.flatnonzero(observations == obs)
        pose = poses[obs]
        camera = predicted_world[rows] @ pose[:3, :3].T + pose[:3, 3]
        distortion = np.array([radial[obs], 0., 0., 0., 0.])
        projected, jac = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), matrices[obs], distortion)
        pose_jacobian[rows]=jac[:,:6].reshape(-1,2,6)
        camera_points[rows]=camera
        target_camera = target_world[rows] @ pose[:3, :3].T + pose[:3, 3]
        target, _ = cv2.projectPoints(target_camera, np.zeros(3), np.zeros(3), matrices[obs], distortion)
        oracle_camera = oracle_world[rows] @ pose[:3, :3].T + pose[:3, 3]
        oracle, _ = cv2.projectPoints(oracle_camera, np.zeros(3), np.zeros(3), matrices[obs], distortion)
        projected = projected.reshape(-1, 2); target = target.reshape(-1, 2)
        residual[rows] = pixels[rows]-projected
        image_error[rows] = pixels[rows]-target
        world_error[rows] = projected-target
        offplane_image_error[rows] = oracle.reshape(-1,2)-target
        j = jac[:, 3:6].reshape(-1, 2, 3) @ pose[:3, :3] @ tangent[rows].transpose(0, 2, 1)
        projected_cov[rows] = (j @ j.transpose(0, 2, 1)) * (uv_variance_scale*prediction[3][rows]).reshape(-1, 1, 1)
    if set(source_views[calibration_rows]) & set(source_views[evaluation_rows]):
        raise ValueError("calibration/evaluation source image leakage")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, residual=residual, image_error=image_error, projected_world_error=world_error,
                        query_variance=variance_scale*prediction[1].reshape(-1), projected_covariance=projected_cov,
                        source_views=source_views, calibration_rows=calibration_rows, evaluation_rows=evaluation_rows,
                        head_content_sha256=np.asarray(head_hash),
                        positive_probability=prediction[-2], negative_probability=prediction[-1],
                        plane_ids=plane_ids, observations=observations,
                        token_ids=(np.floor(token_centers[:,1]/4)*64+np.floor(token_centers[:,0]/4)).astype(np.int64),
                        pose_jacobian=pose_jacobian,camera_points=camera_points,
                        pose_jacobian_semantics=np.asarray('prediction_derivative_left_camera_SE3_rotation_radians_then_translation_metres_at_mapping_pose'),
                        offplane_image_error=offplane_image_error, normal_residual_m=normal_residual_m)


def calibrate(bank):
    c, e = bank["calibration_rows"], bank["evaluation_rows"]
    if len(np.intersect1d(c, e)) or set(bank["source_views"][c]) & set(bank["source_views"][e]):
        raise ValueError("mapping split leakage")
    # Float32 variance multiplication can erase scipy's finite-difference steps.
    r, q, p = (np.asarray(bank[k], np.float64) for k in
               ("residual", "query_variance", "projected_covariance"))
    models = {"independent": (0., 1.)}
    optimizers = {}
    for name, joint in (("scale_only", False), ("correlated", True)):
        def objective(x):
            rho = x[0] if joint else 0.
            return metrics(r[c], covariance(q[c], p[c], rho, np.exp(x[-1])))["nll"]
        result = minimize(objective, np.zeros(2 if joint else 1), method="L-BFGS-B",
                          bounds=([(-.99, .99)] if joint else []) + [(-6., 6.)])
        if not result.success:
            raise RuntimeError(result.message)
        models[name] = (float(result.x[0]) if joint else 0., float(np.exp(result.x[-1])))
        optimizers[name] = str(result.message)
    report = {"head_content_sha256": str(bank["head_content_sha256"]),
              "calibration_count": len(c), "evaluation_count": len(e),
              "query_data_used": False, "optimizer": optimizers, "models": {}}
    for name, (rho, scale) in models.items():
        cov = covariance(q, p, rho, scale)
        report["models"][name] = {"rho": rho, "scale": scale,
            "calibration": metrics(r[c], cov[c]), "evaluation": metrics(r[e], cov[e])}
    for name, joint in (("separate_scales", False), ("joint_separate_scales", True)):
        def separate_objective(x):
            return metrics(r[c], covariance(np.exp(x[0])*q[c], np.exp(x[1])*p[c],
                                            x[2] if joint else 0.))["nll"]
        result = minimize(separate_objective, np.zeros(3 if joint else 2), method="L-BFGS-B",
                          bounds=[(-6.,6.),(-6.,6.)]+([(-.99,.99)] if joint else []))
        if not result.success:
            raise RuntimeError(result.message)
        a, b = np.exp(result.x[:2]); rho = float(result.x[2]) if joint else 0.
        cov = covariance(a*q,b*p,rho)
        report["models"][name] = {"rho":rho,"query_variance_scale":float(a),
            "projected_variance_scale":float(b),"calibration":metrics(r[c],cov[c]),
            "evaluation":metrics(r[e],cov[e]),
            "interior_solution":bool(abs(rho)<.985 and np.max(np.abs(result.x[:2]))<5.99)}
    a, b = bank["image_error"][e], bank["projected_world_error"][e]
    report["evaluation_image_world_error_correlation_xy"] = [float(np.corrcoef(a[:,i], b[:,i])[0,1]) for i in range(2)]
    report["evaluation_reprojection_error_median_p90_px"] = np.percentile(np.linalg.norm(r[e], axis=1), [50,90]).tolist()
    # Correlation must add value beyond a scalar variance recalibration.
    joint = report["models"]["joint_separate_scales"]["evaluation"]
    controls = [report["models"][n]["evaluation"] for n in ("independent", "scale_only", "separate_scales")]
    report["held_mapping_gate_pass"] = bool(all(joint["nll"] < x["nll"] for x in controls)
        and abs(joint["coverage_90"]-.9) <= abs(controls[0]["coverage_90"]-.9)
        and report["models"]["joint_separate_scales"]["interior_solution"])
    report["limitations"] = ["canonical mapping centers, not runtime anonymous modes",
                             "mapping geometry is supervision, not sensor depth truth",
                             "fixed global rho; no deployed match/null prior calibration"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bank, allow_pickle=False) as bank:
        report = calibrate(bank)
    report["bank_sha256"] = hashlib.sha256(args.bank.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
