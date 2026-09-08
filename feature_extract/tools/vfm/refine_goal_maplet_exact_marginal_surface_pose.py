"""Diagnostic exact marginal solve with frozen covariance and physical priors.

Directly optimize the same token mixture NLL used for acceptance. Covariance,
active token set, and token plane-balance weights are frozen at the common
initial pose. This is not yet a joint MoGe/scale solver, and learned match
sigmoids are not asserted to be calibrated probabilities. No GT enters solving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _load, _errors, _threshold_hits, THRESHOLDS
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load as _load_corr, _score
from feature_extract.tools.vfm.fuse_goal_maplet_local_correlation_coordinates import _paired_contract
from feature_extract.tools.vfm.refine_goal_maplet_cross_coordinate_probabilistic_surface_pose import (
    _anchored_component_prior, _candidate_pose,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import (
    _fixed_token_hypotheses, _pose_step, _plane_balance_weights, _image_covariance,
    MAXIMUM_CAMERA_CENTER_STEP_M, MAXIMUM_ROTATION_STEP_DEG,
    MINIMUM_GLOBAL_SUPPORT_FRACTION, MINIMUM_FIXED_ROWS, MINIMUM_PHYSICAL_PLANES,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def _objective(parameter, camera0, pixels, K, k1, inverse_covariance, logdet,
               token_index, prior, existence, token_weights):
    projected, jac = cv2.projectPoints(camera0, parameter[:3], parameter[3:], K,
                                      np.asarray([k1, 0., 0., 0., 0.]))
    residual = projected.reshape(-1, 2) - pixels
    jac = jac[:, :6].reshape(-1, 2, 6)
    weighted = np.einsum("nij,nj->ni", inverse_covariance, residual)
    logmatch = np.log(np.maximum(prior * existence, 1e-300)) - np.log(2 * np.pi)
    logmatch -= .5 * (logdet + np.sum(residual * weighted, axis=1))
    rotation = cv2.Rodrigues(parameter[:3])[0]
    depth = (camera0 @ rotation.T + parameter[3:])[:, 2]
    logmatch[(depth <= 1e-6) | (existence <= 0) | (prior <= 0)] = -np.inf
    count = len(token_weights)
    maximum = np.full(count, -np.inf)
    np.maximum.at(maximum, token_index, logmatch)
    shift = np.where(np.isfinite(maximum), maximum, 0.)
    match_sum = np.bincount(token_index, weights=np.exp(logmatch - shift[token_index]), minlength=count)
    logsum = np.full(count, -np.inf)
    positive = match_sum > 0
    logsum[positive] = np.log(match_sum[positive]) + shift[positive]
    null = np.bincount(token_index, weights=prior * (1 - existence) / (256. * 144.), minlength=count)
    lognull = np.full(count, -np.inf)
    lognull[null > 0] = np.log(null[null > 0])
    total = np.logaddexp(logsum, lognull)
    if not np.isfinite(total).all():
        raise ValueError("mixture has zero density for an observed token")
    responsibility = np.exp(logmatch - total[token_index])
    coefficients = responsibility * token_weights[token_index] / np.sum(token_weights)
    gradient = np.einsum("n,ni,nij->j", coefficients, weighted, jac)
    loss = -float(np.dot(token_weights, total) / np.sum(token_weights))
    return loss, gradient, float(np.sum(responsibility))


def _pose_free_reference(token, plane, prototype, radio_score):
    """Choose plane-balance anchors without consulting camera pose or residuals."""
    order = np.lexsort((np.asarray(prototype), np.asarray(plane), -np.asarray(radio_score), np.asarray(token)))
    sorted_tokens = np.asarray(token)[order]
    return order[np.r_[True, sorted_tokens[1:] != sorted_tokens[:-1]]] if len(order) else order


def _solve(initial, anchor, corr, query, isotropic, *, evaluation_pose=None,
           token_support_policy="initial_nearest"):
    """Solve without GT, or score a supplied pose for a separate post-label audit."""
    lo, hi = map(int, corr["correspondence_offsets"][query:query+2])
    K = np.asarray(corr["camera_matrices"][query], np.float64)
    k1 = float(corr["radial_k1"][query])
    token_all = corr["query_tokens"][lo:hi]
    provenance = corr["provenance_region_plane_atlas_row"][lo:hi]
    if token_support_policy == "initial_nearest":
        reference, _ = _fixed_token_hypotheses(initial, anchor["world_points"][lo:hi], token_all,
                                               anchor["query_measurements_xy"][lo:hi], K, k1)
    elif token_support_policy == "pose_free_radio":
        reference = _pose_free_reference(token_all, provenance[:, 1], anchor["prototype_atlas_row"][lo:hi],
                                         anchor["radio_match_score"][lo:hi])
    else:
        raise ValueError("unknown token support policy")
    planes = provenance[reference, 1]
    detail = {"active_token_count": len(reference), "physical_plane_count": len(np.unique(planes))}
    if len(reference) < MINIMUM_FIXED_ROWS or len(np.unique(planes)) < MINIMUM_PHYSICAL_PLANES:
        return initial.copy(), False, dict(detail, reason="insufficient_support")
    rows = np.flatnonzero(np.isin(token_all, token_all[reference]))
    active_tokens, token_index = np.unique(token_all[rows], return_inverse=True)
    reference_weights = _plane_balance_weights(planes) ** 2
    weight_map = dict(zip(token_all[reference].tolist(), reference_weights.tolist()))
    token_weights = np.asarray([weight_map[int(token)] for token in active_tokens])
    world = np.asarray(corr["world_points"][lo:hi][rows], np.float64)
    pixels = np.asarray(corr["query_measurements_xy"][lo:hi][rows], np.float64)
    purity = np.asarray(corr["prototype_plane_pixel_purity"][lo:hi][rows], np.float64)
    covariance = _image_covariance(initial, world,
        np.asarray(corr["prototype_centroid_covariance_world_m2"][lo:hi][rows], np.float64),
        corr["query_measurement_variance_px2"][lo:hi][rows], purity, K, k1, isotropic)
    prior = _anchored_component_prior(token_all[rows], provenance[rows, 1],
        corr["prototype_atlas_row"][lo:hi][rows], np.zeros(len(rows), np.int64),
        corr["radio_match_score"][lo:hi][rows], policy="radio_gibbs_unit_temperature")
    # These learned sigmoids are diagnostic priors, NOT independently calibrated.
    existence = np.clip(corr["correspondence_match_probability"][lo:hi][rows] * np.clip(purity, 0., 1.),
                        1e-6, 1 - 1e-6)
    camera0 = world @ initial[:3, :3].T + initial[:3, 3]
    inverse_covariance = np.linalg.inv(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    def objective(parameter):
        return _objective(parameter, camera0, pixels, K, k1, inverse_covariance,
                          logdet, token_index, prior, existence, token_weights)
    initial_loss, _, initial_mass = objective(np.zeros(6))
    if evaluation_pose is not None:
        relative_rotation = evaluation_pose[:3, :3] @ initial[:3, :3].T
        parameter = np.r_[cv2.Rodrigues(relative_rotation)[0].reshape(3),
                          evaluation_pose[:3, 3] - relative_rotation @ initial[:3, 3]]
        value, _, mass = objective(parameter)
        return initial.copy(), False, dict(detail, initial_loss=initial_loss,
                                           evaluation_loss=value, evaluation_match_mass=mass)
    bounds = [(-np.deg2rad(MAXIMUM_ROTATION_STEP_DEG), np.deg2rad(MAXIMUM_ROTATION_STEP_DEG))] * 3
    bounds += [(-MAXIMUM_CAMERA_CENTER_STEP_M, MAXIMUM_CAMERA_CENTER_STEP_M)] * 3
    history = [initial_loss]
    solution = minimize(lambda x: objective(x)[:2], np.zeros(6), jac=True, method="L-BFGS-B",
        bounds=bounds, callback=lambda x: history.append(objective(x)[0]),
        options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-7, "maxls": 30})
    output = _candidate_pose(initial, solution.x)
    final_loss, _, final_mass = objective(solution.x)
    angle, distance = _pose_step(initial, output)
    fullworld, fullpixels = corr["world_points"][lo:hi], corr["query_measurements_xy"][lo:hi]
    before = _score(initial, fullworld, token_all, provenance, K, k1, fullpixels)
    after = _score(output, fullworld, token_all, provenance, K, k1, fullpixels)
    support = (after["inlier_count"] >= max(6, int(np.floor(MINIMUM_GLOBAL_SUPPORT_FRACTION * before["inlier_count"])))
               and before["reprojection_median_px"] is not None and after["reprojection_median_px"] is not None
               and after["reprojection_median_px"] <= before["reprojection_median_px"] + .25)
    accepted = bool(solution.success and final_loss < initial_loss - 1e-12
                    and angle <= MAXIMUM_ROTATION_STEP_DEG + 1e-8
                    and distance <= MAXIMUM_CAMERA_CENTER_STEP_M + 1e-8
                    and final_mass >= MINIMUM_GLOBAL_SUPPORT_FRACTION * initial_mass and support)
    detail.update(initial_loss=initial_loss, final_loss=final_loss, loss_history=history,
        solver_success=bool(solution.success), solver_message=str(solution.message),
        initial_match_mass=initial_mass, final_match_mass=final_mass,
        candidate_count=len(rows), rotation_step_deg=angle, center_step_m=distance,
        support_gate_pass=bool(support), accepted=accepted)
    return output if accepted else initial.copy(), accepted, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correspondences", type=Path, nargs=2, required=True)
    parser.add_argument("--initial_poses", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--token_support_policy", choices=("initial_nearest", "pose_free_radio"),
                        default="initial_nearest")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite marginal solve")
    inventories = [_load_corr(path) for path in args.correspondences]
    _paired_contract(inventories[0][0], inventories[0][1], inventories[1][0], inventories[1][1])
    initial, _ = _load(args.initial_poses)
    if not np.array_equal(initial["names"], inventories[0][0]["names"]):
        raise ValueError("pose/correspondence order differs")
    args.output_dir.mkdir(parents=True)
    outputs, details = {}, {}
    for head, (corr, _) in enumerate(inventories):
        for isotropic in (True, False):
            key = f"h{head}_" + ("isotropic" if isotropic else "anisotropic")
            poses = initial["pose_w2c"].copy(); accepted = np.zeros(len(poses), bool); diagnostics = []
            for i in range(len(poses)):
                if initial["usable"][i]:
                    poses[i], accepted[i], detail = _solve(poses[i], inventories[0][0], corr, i, isotropic,
                                                         token_support_policy=args.token_support_policy)
                else:
                    detail = {"reason": "unusable_initial_pose"}
                diagnostics.append(detail)
            arrays = dict(names=initial["names"], pose_w2c=poses, usable=initial["usable"], accepted=accepted)
            meta = dict(artifact_type="goal_maplet_exact_fixed_covariance_marginal_pose_v1",
                arrays_sha256=arrays_sha256(arrays), query_pose_or_ground_truth_read=False,
                production_eligible=False, covariance="isotropic" if isotropic else "anisotropic",
                initial_sha256=file_sha256(args.initial_poses), correspondence_sha256=file_sha256(args.correspondences[head]),
                token_anchor_sha256=file_sha256(args.correspondences[0]),
                match_prior_empirically_calibrated=False, covariance_pose_dependence="frozen_at_initial",
                sparse_dense_moge_joint_factor=False)
            meta["token_support_policy"] = args.token_support_policy
            meta["content_sha256"] = canonical_json_sha256(meta)
            path = args.output_dir / (key + ".npz")
            np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(meta, sort_keys=True)))
            outputs[key] = path; details[key] = diagnostics
    # All four solved inventories are frozen before any query label access.
    summaries = {}
    initial_t, initial_r = _errors(initial, args.query_contributors)
    for key, path in outputs.items():
        arrays, _ = _load(path); t, r = _errors(arrays, args.query_contributors)
        accepted = arrays["accepted"]
        summaries[key] = dict(median_translation_m=float(np.median(t)), median_rotation_deg=float(np.median(r)),
            threshold_hits=[int(np.sum(_threshold_hits(t, r, dt, dr))) for dt, dr in THRESHOLDS],
            accepted_count=int(np.sum(accepted)), accepted_translation_worsened_count=int(np.sum(accepted & (t > initial_t))),
            accepted_rotation_worsened_count=int(np.sum(accepted & (r > initial_r))), file_sha256=file_sha256(path))
    report = dict(artifact_type="goal_maplet_exact_marginal_postlabel_audit_v1", summaries=summaries,
                  diagnostics=details, all_poses_frozen_before_labels=True, production_eligible=False)
    report["content_sha256"] = canonical_json_sha256(report)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
