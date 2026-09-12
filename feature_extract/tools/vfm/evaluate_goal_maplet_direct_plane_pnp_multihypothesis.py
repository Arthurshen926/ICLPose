"""Pose-free grouped PnP hypotheses with a post-freeze development evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    base_keys = (
        "names", "correspondence_offsets", "world_points", "query_tokens",
        "provenance_region_plane_atlas_row", "camera_matrices", "radial_k1",
    )
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        keys = list(base_keys)
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v2",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
        ):
            keys += ["prototype_atlas_row", "query_plane_visible_fraction", "radio_match_score"]
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
        ):
            keys += [
                "prototype_world_covariance_m2", "prototype_plane_pixel_purity",
                "prototype_plane_depth_dispersion_m",
            ]
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
        ):
            keys += ["query_measurements_xy"]
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
        ):
            keys += ["query_measurement_variance_px2", "correspondence_match_probability"]
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
        ):
            keys += ["prototype_centroid_covariance_world_m2"]
        if metadata.get("artifact_type") == "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7":
            keys += [
                "prototype_chart_uv_measurement_m",
                "prototype_chart_uv_cell_lower_m",
            ]
        arrays = {key: np.asarray(data[key]) for key in keys}
    count = len(arrays["names"])
    offsets=arrays['correspondence_offsets'];n=len(arrays['world_points'])
    if (offsets.shape!=(count+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0]!=0 or offsets[-1]!=n or np.any(np.diff(offsets.astype(np.int64))<0)
            or arrays['world_points'].shape!=(n,3) or arrays['query_tokens'].shape!=(n,)
            or arrays['query_tokens'].dtype.kind not in 'iu'
            or np.any(arrays['query_tokens']>=64*36) or np.any(arrays['query_tokens']<0)
            or arrays['radial_k1'].shape!=(count,)):
        raise ValueError('malformed PnP row/token inventory')
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v1",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v2",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
        )
        or metadata.get("pose_or_ground_truth_opened") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or arrays["correspondence_offsets"].shape != (count + 1,)
        or arrays["camera_matrices"].shape != (count, 3, 3)
        or arrays["provenance_region_plane_atlas_row"].shape
        != (len(arrays["world_points"]), 3)
        or (
            "query_measurements_xy" in arrays
            and arrays["query_measurements_xy"].shape != (len(arrays["world_points"]), 2)
        )
        or (
            metadata.get("artifact_type") == "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4"
            and (
                metadata.get("query_measurement_semantics")
                != "centroid_of_observed_same_plane_region_pixels_inside_each_4x4_RADIO_token"
                or metadata.get("hidden_or_occluded_pixels_added_to_query_measurement") != 0
            )
        )
        or (
            metadata.get("artifact_type") == "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5"
            and (
                metadata.get("query_measurement_semantics") not in {
                    "mapping_only_pairwise_RADIO_continuous_subtoken_mean_inside_original_4x4_token",
                    "native_matched_fine_query_pixel_update_inside_original_4x4_token",
                    "mapping_only_RADIO_query_to_anonymous_atlas_prototype_continuous_subtoken_mean_inside_original_4x4_token",
                }
                or metadata.get("query_measurement_uncertainty_semantics")
                != "predicted_isotropic_centroid_measurement_variance_px2_not_surface_footprint"
                or arrays["query_measurement_variance_px2"].shape != (len(arrays["world_points"]),)
                or arrays["correspondence_match_probability"].shape != (len(arrays["world_points"]),)
                or np.any(arrays["query_measurement_variance_px2"] <= 0.0)
                or np.any((arrays["correspondence_match_probability"] < 0.0)
                          | (arrays["correspondence_match_probability"] > 1.0))
            )
        )
        or (
            metadata.get("artifact_type") in {
                "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
                "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
            }
            and (
                metadata.get("query_measurement_semantics")
                != "mapping_only_joint_query_subtoken_and_continuous_chartUV_surface_coordinate_mean"
                or metadata.get("query_measurement_uncertainty_semantics")
                != "predicted_query_centroid_variance_px2_plus_tangent_chartUV_centroid_covariance_world_m2_not_surface_footprint"
                or arrays["query_measurement_variance_px2"].shape != (len(arrays["world_points"]),)
                or arrays["correspondence_match_probability"].shape != (len(arrays["world_points"]),)
                or arrays["prototype_centroid_covariance_world_m2"].shape
                != (len(arrays["world_points"]), 3, 3)
                or np.any(arrays["query_measurement_variance_px2"] <= 0.0)
                or np.any((arrays["correspondence_match_probability"] < 0.0)
                          | (arrays["correspondence_match_probability"] > 1.0))
                or not np.all(np.isfinite(arrays["prototype_centroid_covariance_world_m2"]))
                or (
                    metadata.get("artifact_type")
                    == "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7"
                    and (
                        metadata.get("chart_uv_metric_cell_support_enforced") is not True
                        or arrays.get("prototype_chart_uv_measurement_m", np.zeros((0, 2))).shape
                        != (len(arrays["world_points"]), 2)
                        or arrays.get("prototype_chart_uv_cell_lower_m", np.zeros((0, 2))).shape
                        != (len(arrays["world_points"]), 2)
                        or not np.isfinite(float(metadata.get("chart_uv_metric_cell_size_m", -1.0)))
                        or float(metadata.get("chart_uv_metric_cell_size_m", -1.0)) <= 0.0
                        or not np.all(np.isfinite(arrays["prototype_chart_uv_measurement_m"]))
                        or not np.all(np.isfinite(arrays["prototype_chart_uv_cell_lower_m"]))
                        or not np.array_equal(
                            np.floor(
                                arrays["prototype_chart_uv_measurement_m"]
                                / float(metadata.get("chart_uv_metric_cell_size_m", -1.0))
                            ) * float(metadata.get("chart_uv_metric_cell_size_m", -1.0)),
                            arrays["prototype_chart_uv_cell_lower_m"],
                        )
                        or np.any(
                            arrays["prototype_chart_uv_measurement_m"]
                            < arrays["prototype_chart_uv_cell_lower_m"]
                        )
                        or np.any(
                            arrays["prototype_chart_uv_measurement_m"]
                            >= arrays["prototype_chart_uv_cell_lower_m"]
                            + float(metadata.get("chart_uv_metric_cell_size_m", -1.0))
                        )
                    )
                )
            )
        )
    ):
        raise ValueError("frozen PnP correspondence inventory differs")
    if metadata.get('query_measurement_semantics') == 'native_matched_fine_query_pixel_update_inside_original_4x4_token':
        protocol = metadata.get('fine_readout_protocol', {})
        center = np.c_[(arrays['query_tokens'] % 64)*4+1.5, (arrays['query_tokens']//64)*4+1.5]
        if (protocol.get('anchors_fixed') is not True
                or protocol.get('query_ground_truth_read') is not False
                or metadata.get('fine_readout_arm') not in {'coarse','fine','joint','fine_shrink','fine_calibrated','boundary','boundary_joint','joint_calibrated','fine_calibrated128','reliability_mean','reliability_both','reliability_full','reliability_uniform32','reliability_rank32'}
                or not metadata.get('coarse_correspondence_file_sha256')
                or not metadata.get('frozen_initial_pose_sha256')
                or metadata.get('fine_measurement_variance_recalibrated') not in (False, True)
                or not np.isfinite(arrays['query_measurements_xy']).all()
                or np.any(np.abs(arrays['query_measurements_xy']-center)>2+1e-8)):
            raise ValueError('native fine measurement contract differs')
        if metadata.get('fine_readout_arm') in {'fine_shrink','fine_calibrated','joint_calibrated','fine_calibrated128','reliability_mean','reliability_both','reliability_full','reliability_uniform32','reliability_rank32'}:
            calibration = protocol.get('mapping_calibration', {})
            if (calibration.get('query_ground_truth_read') is not False
                    or calibration.get('heldout_used_to_fit') is not False
                    or not calibration.get('file_sha256')
                    or not 0 <= float(calibration.get('alpha', -1)) <= 1
                    or not np.isfinite(float(calibration.get('variance_scale', np.nan)))
                    or float(calibration.get('variance_scale', 0)) <= 0):
                raise ValueError('native fine calibration contract differs')
        if metadata.get('fine_measurement_variance_recalibrated') is not (metadata.get('fine_readout_arm') in {'fine_calibrated','joint_calibrated','fine_calibrated128','reliability_mean','reliability_both','reliability_full','reliability_uniform32','reliability_rank32'}):
            raise ValueError('native fine variance contract differs')
        if metadata.get('fine_readout_arm','').startswith('reliability_'):
            if (not protocol.get('reliability_model_sha256')
                    or protocol.get('reliability_query_ground_truth_read') is not False
                    or protocol.get('reliability_heldout_used_to_fit') is not False):
                raise ValueError('native fine reliability contract differs')
    for key in ('world_points','camera_matrices','radial_k1','query_measurement_variance_px2','correspondence_match_probability'):
        if key in arrays and not np.isfinite(arrays[key]).all():
            raise ValueError('nonfinite PnP input: '+key)
    return arrays, metadata


def _solve(
    world: np.ndarray,
    tokens: np.ndarray,
    K: np.ndarray,
    k1: float,
    rows: np.ndarray,
    query_measurements_xy: np.ndarray | None = None,
    *, unique_token_lm: bool = True, token_ransac_iterations: int = 0,
    sampling_policy='uniform', hypothesis_budget=None, planes=None, scores=None, sampling_stats=None, guard_lm=False,
) -> np.ndarray | None:
    if token_ransac_iterations:
        from feature_extract.tools.vfm.token_hypothesis_ransac import solve
        return solve(world,tokens,K,k1,rows,query_measurements_xy,iterations=token_ransac_iterations,
                     sampling_policy=sampling_policy,hypothesis_budget=hypothesis_budget,
                     planes=planes,scores=scores,stats=sampling_stats)
    if len(rows) < 6 or (unique_token_lm and len(np.unique(tokens[rows])) < 6):
        return None
    pixel_all = (
        np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        if query_measurements_xy is None
        else np.asarray(query_measurements_xy, np.float64).reshape(-1, 2)
    )
    if len(pixel_all) != len(tokens) or not np.all(np.isfinite(pixel_all)):
        raise ValueError("query measurements differ from tokens")
    pixel = pixel_all[rows]
    distortion = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64)
    cv2.setRNGSeed(260901)
    ok, rvec, tvec, inlier = cv2.solvePnPRansac(
        world[rows].astype(np.float64), pixel.astype(np.float64), K, distortion,
        iterationsCount=1000, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inlier is None or len(inlier) < 6:
        return None
    selected = rows[inlier.reshape(-1)]
    if unique_token_lm:
        camera=world[selected]@cv2.Rodrigues(rvec)[0].T+np.asarray(tvec).reshape(3)
        selected=selected[np.isfinite(camera).all(axis=1)&(camera[:,2]>1e-8)]
        if len(selected)<6:return None
        projected,_=cv2.projectPoints(world[selected],rvec,tvec,K,distortion)
        residual=np.linalg.norm(projected.reshape(-1,2)-pixel_all[selected],axis=1)
        selected=_unique_token_rows(selected,tokens,residual)
        if len(selected)<6:return None
    all_pixel = pixel_all[selected]
    initial=np.eye(4);initial[:3,:3]=cv2.Rodrigues(rvec)[0];initial[:3,3]=np.asarray(tvec).reshape(3)
    rvec, tvec = cv2.solvePnPRefineLM(
        world[selected], all_pixel, K, distortion, rvec, tvec,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = np.asarray(tvec).reshape(3)
    if guard_lm:
        from feature_extract.tools.vfm.token_hypothesis_ransac import score_pose
        local_tokens=tokens[rows];groups=[np.flatnonzero(local_tokens==t) for t in np.unique(local_tokens)]
        before=score_pose(initial,world[rows],pixel_all[rows],groups,K,k1)[0]
        after=score_pose(pose,world[rows],pixel_all[rows],groups,K,k1)[0] if np.isfinite(pose).all() else (-1,-np.inf)
        accepted=after>before
        if sampling_stats is not None:sampling_stats.update(lm_accepted=accepted,initial_unique_support=before[0],refined_unique_support=after[0])
        return pose if accepted else initial
    return pose if np.isfinite(pose).all() else None


def _unique_token_rows(rows,tokens,residual):
    rows=np.asarray(rows,np.int64);residual=np.asarray(residual,float)
    if residual.shape!=rows.shape or not np.isfinite(residual).all():
        raise ValueError('invalid token association residual')
    order=np.lexsort((rows,residual,tokens[rows]))
    _,first=np.unique(tokens[rows[order]],return_index=True)
    return rows[order[first]]


def _select_match_rows(tokens,provenance,scores,per_plane=False):
    group=tokens
    if per_plane:
        _,group=np.unique(np.c_[tokens,provenance[:,1]],axis=0,return_inverse=True)
    return _unique_token_rows(np.arange(len(tokens)),group,-np.asarray(scores,float))


def _score(
    pose: np.ndarray,
    world: np.ndarray,
    tokens: np.ndarray,
    provenance: np.ndarray,
    K: np.ndarray,
    k1: float,
    query_measurements_xy: np.ndarray | None = None,
) -> dict[str, object]:
    pixel = (
        np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        if query_measurements_xy is None
        else np.asarray(query_measurements_xy, np.float64).reshape(-1, 2)
    )
    if len(pixel) != len(tokens) or not np.all(np.isfinite(pixel)):
        raise ValueError("query measurements differ from tokens")
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], K,
        np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    valid = (camera[:, 2] > 0.0) & (residual <= 4.0)
    # A query token may carry multiple mutually exclusive 3D hypotheses.  A
    # candidate pose can receive support from that image location only once:
    # retain the valid hypothesis with the smallest residual.
    chosen = []
    for token_id in np.unique(tokens):
        rows = np.flatnonzero((tokens == token_id) & valid)
        if len(rows):
            chosen.append(int(rows[np.argmin(residual[rows])]))
    inlier = np.asarray(chosen, np.int64)

    def support(column: int, cap: int) -> tuple[int, int]:
        if not len(inlier):
            return 0, 0
        _, count = np.unique(provenance[inlier, column], return_counts=True)
        return int(np.sum(np.minimum(count, cap))), int(np.sum(count >= 3))

    region_capped, region_supported = support(0, 8)
    plane_capped, plane_supported = support(1, 8)
    view_capped, view_supported = support(2, 8)
    return {
        "pose": pose,
        "inlier_count": int(len(inlier)),
        "inlier_ratio": float(len(inlier) / max(len(np.unique(tokens)), 1)),
        "region_capped_support": region_capped,
        "plane_capped_support": plane_capped,
        "view_capped_support": view_capped,
        "supported_region_count": region_supported,
        "supported_plane_count": plane_supported,
        "supported_view_count": view_supported,
        "reprojection_median_px": None if not len(inlier) else float(np.median(residual[inlier])),
    }


def _top_groups(values: np.ndarray, maximum: int, tokens: np.ndarray | None = None) -> list[np.ndarray]:
    groups = []
    for value in np.unique(values):
        rows = np.flatnonzero(values == value)
        support=len(rows) if tokens is None else len(np.unique(tokens[rows]))
        if support >= 6:
            groups.append(rows)
    groups.sort(key=lambda rows: (-(len(rows) if tokens is None else len(np.unique(tokens[rows]))), int(values[rows[0]]) if tokens is not None else int(rows[0])))
    return groups[:maximum]


def _choice_key(candidate: dict[str, object], rule: str) -> tuple[object, ...]:
    residual=candidate['reprojection_median_px']
    penalty=float(residual) if residual is not None and np.isfinite(residual) else 1e9
    if rule == "raw_inliers":
        return (candidate["inlier_count"], -penalty)
    if rule == "balanced_support":
        return (
            candidate["region_capped_support"], candidate["plane_capped_support"],
            candidate["view_capped_support"], candidate["inlier_count"],
            -penalty,
        )
    if rule == "supported_entities":
        return (
            candidate["supported_region_count"], candidate["supported_plane_count"],
            candidate["supported_view_count"], candidate["inlier_count"],
            -penalty,
        )
    raise ValueError("unknown selection rule")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--maximum_groups_per_kind", type=int, default=16)
    parser.add_argument("--seed_group_support",choices=('row_count','unique_tokens'),default='unique_tokens')
    parser.add_argument("--output_frozen_candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver_policy",choices=('legacy','unique_token_lm','unique_token_guarded_lm','token_ransac'),default='unique_token_lm')
    parser.add_argument("--token_ransac_iterations",type=int,default=128)
    parser.add_argument('--token_sampling_policy',choices=('uniform','geometry','geometry_score'),default='uniform')
    parser.add_argument('--token_hypothesis_budget',type=int)
    parser.add_argument("--association_policy",choices=('all','match_top1','match_top1_per_plane','mapping_gate'),default='all')
    parser.add_argument("--mapping_match_gate",type=Path)
    parser.add_argument("--physical_regions",type=Path,help="Opt-in frozen physical centers/radius for plane-group replacement or augmentation.")
    parser.add_argument("--seed_group_reference",type=Path,help="Optional original correspondence inventory fixes per-query seed-group call budgets during augmentation.")
    parser.add_argument("--physical_region_mode",choices=["replace","augment"],default="replace")
    parser.add_argument('--global_proposal_sampler',choices=['cv','token','row','token_prior'],default='cv')
    parser.add_argument('--global_sampling_prior',type=Path)
    parser.add_argument('--global_proposal_budget',type=int,default=1000)
    parser.add_argument('--global_proposal_attempts',type=int,default=5000)
    args = parser.parse_args()
    if args.global_proposal_budget<1 or args.global_proposal_attempts<1:
        raise ValueError('positive global proposal limits required')
    if args.global_proposal_sampler!='cv' and args.solver_policy!='unique_token_guarded_lm':
        raise ValueError('global proposal experiment requires fixed guarded backend')
    if args.output.exists() or args.output_frozen_candidates.exists():
        raise FileExistsError("refusing to overwrite multi-hypothesis result")
    arrays, upstream = _load(args.frozen_correspondences)
    if args.token_sampling_policy=='geometry_score' and (
            args.association_policy!='all' or 'correspondence_match_probability' not in arrays):
        raise ValueError('score-guided experiment requires full candidates and stored match scores')
    sampling_prior = None
    if args.global_proposal_sampler == 'token_prior':
        if args.global_sampling_prior is None or args.association_policy != 'all':
            raise ValueError('context prior requires full correspondence inventory and prior')
        from feature_extract.tools.vfm.region_sampling_prior import load_region_sampling_prior
        sampling_prior = load_region_sampling_prior(args.global_sampling_prior, args.frozen_correspondences, arrays)
    elif args.global_sampling_prior is not None:
        raise ValueError('unused global sampling prior')
    reference = None
    if args.seed_group_reference:
        if args.association_policy != 'all':
            raise ValueError('reference budgets require unfiltered augmentation')
        reference, _ = _load(args.seed_group_reference)
        for key in ('names', 'camera_matrices', 'radial_k1'):
            if not np.array_equal(reference[key], arrays[key]):
                raise ValueError('seed reference query/camera inventory differs')
        # Enforce the contract that every original row survives as a query prefix.
        for i in range(len(arrays['names'])):
            a, b = map(int, reference['correspondence_offsets'][i:i+2])
            c, d = map(int, arrays['correspondence_offsets'][i:i+2])
            for key in ('world_points', 'query_tokens', 'provenance_region_plane_atlas_row'):
                if d-c < b-a or not np.array_equal(reference[key][a:b], arrays[key][c:c+b-a]):
                    raise ValueError('seed reference is not an unchanged query prefix')
    physical_centers=None;physical_radius=None
    if args.physical_regions:
        with np.load(args.physical_regions) as z:
            if 'member_policy' in z and str(z['member_policy'])!='sphere':
                raise ValueError('explicit point membership requires the prototype-aware structured-memory evaluator; this correspondence interface only supports metric envelopes')
            physical_centers=z["centers"];physical_radius=np.asarray(z['radii'] if 'radii' in z else z['radius'],float)
            if 'training_images' in z and set(z['training_images'].astype(str))&set(arrays['names'].astype(str)):raise ValueError('boundary training/test overlap')
        if physical_radius.ndim>1 or (physical_radius.ndim==1 and len(physical_radius)!=len(physical_centers)):raise ValueError('boundary inventory differs')
        if physical_centers.ndim!=2 or physical_centers.shape[1]!=3 or not np.isfinite(physical_centers).all() or not np.isfinite(physical_radius).all() or np.any(physical_radius<=0):raise ValueError("invalid physical regions")
    gate=None
    if args.association_policy=='mapping_gate':
        if args.mapping_match_gate is None:raise ValueError('mapping gate required')
        gate=json.loads(args.mapping_match_gate.read_text())
        if (gate.get('head_content_sha256')!=upstream.get('mapping_subtoken_head_content_sha256')
                or gate.get('query_data_used') is not False or not 0<=float(gate['threshold'])<=1):
            raise ValueError('mapping gate lineage differs')
    rules = ("raw_inliers", "balanced_support", "supported_entities")
    selected: dict[str, list[dict[str, object]]] = {rule: [] for rule in rules}
    all_candidates: list[list[dict[str, object]]] = []
    diagnostic_rows = []
    sampling_records=[]
    for query_index, name in enumerate(arrays["names"].astype(str).tolist()):
        lo, hi = map(int, arrays["correspondence_offsets"][query_index:query_index + 2])
        world = arrays["world_points"][lo:hi]
        tokens = arrays["query_tokens"][lo:hi]
        measurements = arrays["query_measurements_xy"][lo:hi] if "query_measurements_xy" in arrays else None
        provenance = arrays["provenance_region_plane_atlas_row"][lo:hi]
        K = arrays["camera_matrices"][query_index]
        k1 = float(arrays["radial_k1"][query_index])
        score_world,score_tokens,score_provenance,score_measurements=world,tokens,provenance,measurements
        if args.association_policy!='all':
            scores=arrays['correspondence_match_probability'][lo:hi]
            retained=(np.flatnonzero(scores>=float(gate['threshold'])) if gate is not None else
                      _select_match_rows(tokens,provenance,scores,args.association_policy=='match_top1_per_plane'))
            world,tokens,provenance=world[retained],tokens[retained],provenance[retained]
            if measurements is not None:measurements=measurements[retained]
        seed_groups: list[tuple[str, np.ndarray]] = [("all", np.arange(len(world)))]
        third_label=('atlas_prototype' if upstream.get('correspondence_semantics')=='query_RADIO_to_view_independent_metric_plane_UV_texels' else 'source_view')
        for label, column in (("plane", 1), (third_label, 2), ("query_region", 0)):
            maximum = args.maximum_groups_per_kind
            if reference is not None:
                rlo, rhi = map(int, reference['correspondence_offsets'][query_index:query_index+2])
                maximum = len(_top_groups(reference['provenance_region_plane_atlas_row'][rlo:rhi, column],
                    maximum, reference['query_tokens'][rlo:rhi] if args.seed_group_support=='unique_tokens' else None))
            local_groups=_top_groups(provenance[:, column], maximum,
                                     tokens if args.seed_group_support=='unique_tokens' else None)
            if label=='plane' and physical_centers is not None:
                from feature_extract.vfm.localization_goal_maplet.metric_region_memory import activate_regions
                # Uniform token evidence: no uncalibrated cross-plane score comparison.
                regional,_=activate_regions(world,tokens,np.zeros(len(tokens)),physical_centers,physical_radius,len(local_groups),False)
                if args.physical_region_mode=='augment':seed_groups.extend(('plane',rows) for rows in local_groups)
                seed_groups.extend(('physical_region',rows) for rows in regional)
                if args.physical_region_mode=='replace':seed_groups.extend(('plane_fallback',rows) for rows in local_groups[:len(local_groups)-len(regional)])
            else:seed_groups.extend((label,rows) for rows in local_groups)
        candidates = []
        for origin, rows in seed_groups:
            sampling_stats={'query_index':query_index,'origin':origin}
            global_override = origin=='all' and args.global_proposal_sampler!='cv'
            pose = _solve(world, tokens, K, k1, rows, measurements,
                          unique_token_lm=args.solver_policy!='legacy',
                          guard_lm=args.solver_policy=='unique_token_guarded_lm',
                          token_ransac_iterations=(args.global_proposal_attempts if global_override else args.token_ransac_iterations if args.solver_policy=='token_ransac' else 0),
                          sampling_policy=({'token':'uniform','row':'row_uniform','token_prior':'context_prior'}[args.global_proposal_sampler] if global_override else args.token_sampling_policy),
                          hypothesis_budget=args.global_proposal_budget if global_override else args.token_hypothesis_budget,
                          planes=provenance[:,1],
                          scores=(sampling_prior[lo:hi] if global_override and sampling_prior is not None else arrays['correspondence_match_probability'][lo:hi]
                                  if args.association_policy=='all' and 'correspondence_match_probability' in arrays else None),
                          sampling_stats=sampling_stats)
            if args.solver_policy in ('token_ransac','unique_token_guarded_lm'):sampling_records.append(sampling_stats)
            if pose is None:
                continue
            score = _score(pose, score_world, score_tokens, score_provenance, K, k1, score_measurements)
            score["origin"] = origin
            candidates.append(score)
        if not candidates:
            candidates.append({
                "pose": np.full((4, 4), np.nan), "origin": "none", "inlier_count": 0,
                "inlier_ratio": 0.0, "region_capped_support": 0,
                "plane_capped_support": 0, "view_capped_support": 0,
                "supported_region_count": 0, "supported_plane_count": 0,
                "supported_view_count": 0, "reprojection_median_px": None,
            })
        all_candidates.append(candidates)
        row = {"name": name, "candidate_count": int(len(candidates))}
        for rule in rules:
            best = max(candidates, key=lambda candidate: _choice_key(candidate, rule))
            selected[rule].append(best)
            row[rule] = {key: value for key, value in best.items() if key != "pose"}
        diagnostic_rows.append(row)

    candidate_offsets = np.zeros(len(all_candidates) + 1, np.int64)
    for index, candidates in enumerate(all_candidates):
        candidate_offsets[index + 1] = candidate_offsets[index] + len(candidates)
    frozen_arrays: dict[str, np.ndarray] = {
        "names": arrays["names"],
        "candidate_offsets": candidate_offsets,
        "candidate_pose_w2c": np.asarray([
            candidate["pose"] for candidates in all_candidates for candidate in candidates
        ], np.float64),
        "candidate_origin": np.asarray([
            candidate["origin"] for candidates in all_candidates for candidate in candidates
        ]),
        "candidate_inlier_count": np.asarray([
            candidate["inlier_count"] for candidates in all_candidates for candidate in candidates
        ], np.int64),
    }
    for rule in rules:
        frozen_arrays[f"{rule}_pose_w2c"] = np.asarray([row["pose"] for row in selected[rule]])
        frozen_arrays[f"{rule}_inlier_count"] = np.asarray([row["inlier_count"] for row in selected[rule]])
        frozen_arrays[f"{rule}_inlier_ratio"] = np.asarray(
            [row["inlier_ratio"] for row in selected[rule]], np.float64
        )
    frozen_metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_grouped_multihypothesis_v1",
        "arrays_sha256": arrays_sha256(frozen_arrays),
        "query_count": int(len(arrays["names"])),
        "candidate_generation": "all plus grouped physical-plane/third-provenance/query-region PnP seeds; optional physical-region replacement or augmentation of plane groups",
        "physical_regions_sha256":file_sha256(args.physical_regions) if args.physical_regions else None,
        "physical_region_radius":physical_radius.tolist() if physical_radius is not None else None,
        "physical_region_mode":args.physical_region_mode if args.physical_regions else None,
        "third_provenance_semantics":('anonymous_atlas_prototype_not_source_view'
            if upstream.get('correspondence_semantics')=='query_RADIO_to_view_independent_metric_plane_UV_texels' else 'upstream_source_view_or_atlas_row'),
        "global_proposal_sampler":args.global_proposal_sampler,
        "global_proposal_budget":args.global_proposal_budget if args.global_proposal_sampler!="cv" else None,
        "global_proposal_attempts":args.global_proposal_attempts if args.global_proposal_sampler!="cv" else None,
        "global_proposal_scope":"only all-correspondence proposal; other seed solvers unchanged; AP3P scored-model budget differs from OpenCV adaptive iteration cap",
        "solver_policy":args.solver_policy,"association_policy":args.association_policy,
        "seed_group_reference_sha256":file_sha256(args.seed_group_reference) if args.seed_group_reference else None,
        "seed_budget_semantics":("reference group counts; global AP3P uses separately declared scored-model and attempt budgets" if args.global_proposal_sampler!="cv" else "matched group call counts and maximum iterations, not adaptive iterations or wall time" if reference is not None else "native inventory group counts"),
        "seed_group_support":args.seed_group_support,
        "token_ransac_iterations":args.token_ransac_iterations if args.solver_policy=='token_ransac' else None,
        'global_sampling_prior_file_sha256':file_sha256(args.global_sampling_prior) if args.global_sampling_prior else None,
        'token_sampling_policy':args.token_sampling_policy,
        'token_sampling_implementation':'plane_first_spatial_score_token_priority_v2_shared_rng',
        'token_hypothesis_budget':args.token_hypothesis_budget,
        'sampling_records':sampling_records,
        "mapping_match_gate":gate,
        "candidate_scoring_population":"all_original_correspondences_unique_token_support",
        "legacy_view_support_fields":"third_provenance_support_not_independent_view_evidence_for_metric_atlas",
        "selection_rules": list(rules),
        "query_pose_or_ground_truth_opened": False,
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": upstream.get("content_sha256"),
    }
    frozen_metadata["content_sha256"] = canonical_json_sha256(frozen_metadata)
    args.output_frozen_candidates.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_candidates, **frozen_arrays,
        metadata_json=np.asarray(json.dumps(frozen_metadata, sort_keys=True)),
    )

    summaries = {}
    postlabel_rows = []
    for index, name in enumerate(arrays["names"].astype(str).tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        gt_center = -gt[:3, :3].T @ gt[:3, 3]
        row = {"name": name}
        for rule in rules:
            pose = np.asarray(selected[rule][index]["pose"], np.float64)
            if np.all(np.isfinite(pose)):
                center = -pose[:3, :3].T @ pose[:3, 3]
                translation = float(np.linalg.norm(center - gt_center))
                rotation = float(
                    Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude()
                    * 180.0 / np.pi
                )
            else:
                translation = rotation = float("inf")
            row[rule] = {"translation_error_m": translation, "rotation_error_deg": rotation}
        candidate_errors = []
        for candidate in all_candidates[index]:
            pose = np.asarray(candidate["pose"], np.float64)
            if not np.all(np.isfinite(pose)):
                continue
            center = -pose[:3, :3].T @ pose[:3, 3]
            candidate_errors.append((
                float(np.linalg.norm(center - gt_center)),
                float(
                    Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude()
                    * 180.0 / np.pi
                ),
            ))
        row["candidate_oracle_2m45"] = bool(any(
            translation <= 2.0 and rotation <= 45.0
            for translation, rotation in candidate_errors
        ))
        row["candidate_oracle_1m10"] = bool(any(
            translation <= 1.0 and rotation <= 10.0
            for translation, rotation in candidate_errors
        ))
        row["minimum_candidate_translation_m"] = (
            None if not candidate_errors else min(value[0] for value in candidate_errors)
        )
        postlabel_rows.append(row)
    for rule in rules:
        values = [row[rule] for row in postlabel_rows]
        summaries[rule] = {
            "recall_2m45": float(np.mean([
                value["translation_error_m"] <= 2.0 and value["rotation_error_deg"] <= 45.0
                for value in values
            ])),
            "recall_1m10": float(np.mean([
                value["translation_error_m"] <= 1.0 and value["rotation_error_deg"] <= 10.0
                for value in values
            ])),
            "median_translation_m": float(np.median([value["translation_error_m"] for value in values])),
            "median_rotation_deg": float(np.median([value["rotation_error_deg"] for value in values])),
        }
    summaries["candidate_pool_oracle"] = {
        "recall_2m45": float(np.mean([row["candidate_oracle_2m45"] for row in postlabel_rows])),
        "recall_1m10": float(np.mean([row["candidate_oracle_1m10"] for row in postlabel_rows])),
        "note": "post-label upper bound over poses frozen before labels; not deployable selection",
    }
    report = {
        "artifact_type": "goal_maplet_direct_plane_pnp_grouped_multihypothesis_dev_evaluation_v1",
        "candidate_inventory_file_sha256": file_sha256(args.output_frozen_candidates),
        "candidate_inventory_content_sha256": frozen_metadata["content_sha256"],
        "selection_frozen_before_query_pose_opened": True,
        "summaries": summaries,
        "pose_free_diagnostics": diagnostic_rows,
        "postlabel_rows": postlabel_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("pose_free_diagnostics", "postlabel_rows")}, indent=2))


if __name__ == "__main__":
    main()
