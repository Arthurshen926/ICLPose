"""Post-label 2x2x2 diagnostic; freeze every pose before loading any query labels.

Factors are upstream initialization, coordinate means, and coordinate covariance.
Within each initialization, V11 nearest-reprojection row IDs and covariance
projection Jacobians stay fixed. This isolates a local solve, not the production
pipeline's changing row selection/global-support acceptance policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _load, _errors, _threshold_hits, THRESHOLDS
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load as _load_corr
from feature_extract.tools.vfm.fuse_goal_maplet_local_correlation_coordinates import _paired_contract
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import (
    _fixed_token_hypotheses, _fixed_reprojection_sigma_px, _refine_pose,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def _factor_inputs(baseline, local, means, covariance):
    mean_source = (baseline, local)[means]
    cov_source = (baseline, local)[covariance]
    return (mean_source["world_points"], mean_source["query_measurements_xy"],
            cov_source["query_measurement_variance_px2"],
            cov_source["prototype_centroid_covariance_world_m2"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_correspondences", type=Path, required=True)
    parser.add_argument("--local_correspondences", type=Path, required=True)
    parser.add_argument("--initial_poses", type=Path, nargs="+", required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.initial_poses) not in (1,2):
        raise ValueError("provide one or two frozen initial inventories")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite diagnostic experiment")
    baseline, bm = _load_corr(args.baseline_correspondences)
    local, lm = _load_corr(args.local_correspondences)
    _paired_contract(baseline, bm, local, lm)
    initial = [_load(path)[0] for path in args.initial_poses]
    if any(not np.array_equal(p["names"], baseline["names"]) for p in initial):
        raise ValueError("initial pose query order differs")
    args.output_dir.mkdir(parents=True)
    inventory = {}
    for init_index, poses in enumerate(initial):
        frozen_rows = []
        sigma_options = []
        for i, pose in enumerate(poses["pose_w2c"]):
            lo, hi = map(int, baseline["correspondence_offsets"][i:i+2])
            if not poses["usable"][i]:
                frozen_rows.append(np.zeros(0, np.int64)); sigma_options.append([np.zeros(0)] * 2)
                continue
            K, k1 = baseline["camera_matrices"][i], float(baseline["radial_k1"][i])
            rows, _ = _fixed_token_hypotheses(pose, baseline["world_points"][lo:hi],
                baseline["query_tokens"][lo:hi], baseline["query_measurements_xy"][lo:hi], K, k1)
            frozen_rows.append(rows)
            sigmas = []
            for cov_source in (baseline, local):
                sigma = _fixed_reprojection_sigma_px(pose, baseline["world_points"][lo:hi],
                    baseline["prototype_world_covariance_m2"][lo:hi],
                    baseline["prototype_plane_pixel_purity"][lo:hi],
                    baseline["prototype_plane_depth_dispersion_m"][lo:hi], K,
                    query_measurement_variance_px2=cov_source["query_measurement_variance_px2"][lo:hi],
                    centroid_covariance_world_m2=cov_source["prototype_centroid_covariance_world_m2"][lo:hi],
                    include_map_footprint_scatter=False, radial_k1=k1)
                sigmas.append(sigma[rows])
            sigma_options.append(sigmas)
        offsets = np.r_[0, np.cumsum([len(rows) for rows in frozen_rows])]
        selected = np.concatenate([rows + int(baseline["correspondence_offsets"][i])
                                   for i, rows in enumerate(frozen_rows)])
        for means in (0, 1):
            for covariance in (0, 1):
                world, pixel, _, _ = _factor_inputs(baseline, local, means, covariance)
                output = poses["pose_w2c"].copy()
                accepted = np.zeros(len(output), bool)
                for i, pose in enumerate(poses["pose_w2c"]):
                    if not poses["usable"][i]:
                        continue
                    rows = selected[offsets[i]:offsets[i+1]]
                    updated, use, _ = _refine_pose(pose, world[rows], pixel[rows],
                        baseline["provenance_region_plane_atlas_row"][rows, 1], sigma_options[i][covariance],
                        baseline["camera_matrices"][i], float(baseline["radial_k1"][i]))
                    if use:
                        output[i] = updated
                        accepted[i] = True
                arrays = dict(names=poses["names"], pose_w2c=output, usable=poses["usable"],
                              accepted=accepted, fixed_row_offsets=offsets, fixed_global_rows=selected)
                meta = dict(artifact_type="goal_maplet_fixed_row_factor_pose_diagnostic_v1",
                    arrays_sha256=arrays_sha256(arrays), query_pose_or_ground_truth_read=False,
                    initialization_index=init_index, mean_source=means, covariance_source=covariance,
                    row_selection_source="baseline_at_fixed_initial_pose", jacobian_source="baseline_world_at_fixed_initial_pose",
                    global_support_acceptance_applied=False, production_eligible=False,
                    initial_file_sha256=file_sha256(args.initial_poses[init_index]),
                    baseline_correspondence_sha256=file_sha256(args.baseline_correspondences),
                    local_correspondence_sha256=file_sha256(args.local_correspondences))
                meta["content_sha256"] = canonical_json_sha256(meta)
                key = f"i{init_index}_m{means}_c{covariance}"
                path = args.output_dir / (key + ".npz")
                np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(meta, sort_keys=True)))
                inventory[key] = path
    # All output poses are now on disk; only now open contributor GT.
    summaries = {}
    for key, path in inventory.items():
        arrays, _ = _load(path)
        t, r = _errors(arrays, args.query_contributors)
        summaries[key] = dict(median_translation_m=float(np.median(t)), median_rotation_deg=float(np.median(r)),
            threshold_hits=[int(np.sum(_threshold_hits(t, r, dt, dr))) for dt, dr in THRESHOLDS],
            accepted_count=int(np.sum(arrays["accepted"])), file_sha256=file_sha256(path))
    report = dict(artifact_type="goal_maplet_fixed_row_factor_audit_v1", summaries=summaries,
                  all_output_poses_frozen_before_labels=True, output_pose_arm_count=len(inventory), production_eligible=False,
                  limitation="conditional_local_solve;not_production_global_acceptance_or_new_route_test")
    report["content_sha256"] = canonical_json_sha256(report)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
