"""Refine a surface-atlas pose by robust MoGe3-to-atlas Sim(3) registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def _token_points(points: np.ndarray, valid: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((len(tokens), 3), np.float64); keep = np.zeros(len(tokens), bool)
    for row, token in enumerate(np.asarray(tokens, np.int64)):
        y, x = divmod(int(token), 64)
        mask = valid[y * 4 : y * 4 + 4, x * 4 : x * 4 + 4]
        block = points[y * 4 : y * 4 + 4, x * 4 : x * 4 + 4]
        if np.any(mask): output[row] = np.median(block[mask], axis=0); keep[row] = True
    return output, keep


def _unique_reprojection_matches(pose: np.ndarray, world: np.ndarray, tokens: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    pixel = np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
    projected, _ = cv2.projectPoints(world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], K, np.asarray([k1, 0, 0, 0, 0], np.float64))
    error = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    valid = (camera[:, 2] > 0) & np.isfinite(error) & (error <= 4.0)
    chosen = []
    for token in np.unique(tokens):
        rows = np.flatnonzero((tokens == token) & valid)
        if len(rows): chosen.append(int(rows[np.argmin(error[rows])]))
    return np.asarray(chosen, np.int64)


def _fit_surface_sim3(pose_w2c: np.ndarray, query_camera: np.ndarray, world: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
    pose = np.asarray(pose_w2c, np.float64); query = np.asarray(query_camera, np.float64); world = np.asarray(world, np.float64)
    if len(world) < 20: return pose.copy(), 1.0, False, {"match_count": float(len(world))}
    Rcw = pose[:3, :3].T; center = -Rcw @ pose[:3, 3]
    rotated = query @ Rcw.T
    denominator = np.sum(rotated * rotated, axis=1)
    scale_rows = np.sum((world - center) * rotated, axis=1) / np.maximum(denominator, 1e-12)
    positive = scale_rows[np.isfinite(scale_rows) & (scale_rows > 0)]
    if len(positive) < 20: return pose.copy(), 1.0, False, {"match_count": float(len(world))}
    scale0 = float(np.median(positive))

    def components(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        rotation = Rotation.from_rotvec(parameter[:3]).as_matrix() @ Rcw
        translation = center + parameter[3:6]
        scale = scale0 * float(np.exp(parameter[6]))
        predicted = scale * (query @ rotation.T) + translation
        return predicted, rotation, scale

    initial = np.zeros(7, np.float64)
    initial_error = np.linalg.norm(components(initial)[0] - world, axis=1)
    solution = least_squares(lambda p: ((components(p)[0] - world) / 0.20).reshape(-1), initial, loss="huber", f_scale=1.0, max_nfev=100)
    predicted, rotation, scale = components(solution.x)
    final_error = np.linalg.norm(predicted - world, axis=1)
    rotation_change = float(Rotation.from_matrix(rotation @ Rcw.T).magnitude() * 180.0 / np.pi)
    translation_change = float(np.linalg.norm(solution.x[3:6]))
    accepted = bool(
        solution.success and 0.25 <= scale <= 4.0 and rotation_change <= 10.0
        and translation_change <= 2.0
        and np.median(final_error) < np.median(initial_error)
        and np.quantile(final_error, 0.9) <= np.quantile(initial_error, 0.9)
    )
    output = pose.copy()
    if accepted:
        output[:3, :3] = rotation.T; output[:3, 3] = -rotation.T @ (center + solution.x[3:6])
    return output, scale, accepted, {
        "match_count": float(len(world)), "initial_median_m": float(np.median(initial_error)),
        "final_median_m": float(np.median(final_error)), "initial_p90_m": float(np.quantile(initial_error, .9)),
        "final_p90_m": float(np.quantile(final_error, .9)), "rotation_change_deg": rotation_change,
        "translation_change_m": translation_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_pose_inventory", type=Path, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, nargs=2, required=True)
    parser.add_argument("--moge3_dir", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists(): raise FileExistsError("refusing to overwrite surface registration")
    with np.load(args.selected_pose_inventory, allow_pickle=False) as data:
        pose_meta = json.loads(str(data["metadata_json"].item())); pose_all = {k: np.asarray(data[k]) for k in data.files if k != "metadata_json"}
    if pose_meta.get("artifact_type") != "goal_maplet_dual_surface_geometry_consensus_selected_v1" or pose_meta.get("query_pose_or_ground_truth_read") is not False or arrays_sha256(pose_all) != pose_meta.get("arrays_sha256"):
        raise ValueError("selected surface pose contract differs")
    corr = [_load(path)[0] for path in args.frozen_correspondences]
    names = pose_all["names"].astype(str); output_pose = pose_all["pose_w2c"].copy(); accepted = np.zeros(len(names), bool); scales = np.ones(len(names)); match_count = np.zeros(len(names), np.int64); diagnostics=[]
    for index, name in enumerate(names):
        branch = int(pose_all["selected_branch"][index]); value = corr[branch]
        lo, hi = map(int, value["correspondence_offsets"][index:index+2]); world=value["world_points"][lo:hi]; tokens=value["query_tokens"][lo:hi]
        rows = _unique_reprojection_matches(output_pose[index], world, tokens, value["camera_matrices"][index], float(value["radial_k1"][index]))
        with np.load(args.moge3_dir / name, allow_pickle=False) as data:
            query, valid = _token_points(np.asarray(data["points_camera"], np.float64), np.asarray(data["valid"], bool), tokens[rows])
        refined, scale, ok, row = _fit_surface_sim3(output_pose[index], query[valid], world[rows][valid])
        output_pose[index]=refined; scales[index]=scale; accepted[index]=ok; match_count[index]=int(np.sum(valid)); diagnostics.append(row)
    arrays={"names":names,"pose_w2c":output_pose,"usable":pose_all["usable"],"surface_registration_accepted":accepted,"moge3_surface_scale":scales,"surface_match_count":match_count}
    metadata={"artifact_type":"goal_maplet_moge3_to_anonymous_plane_surface_registration_v1","query_pose_or_ground_truth_read":False,"source_rgb_stored_or_consumed_at_runtime":False,"selected_pose_file_sha256":file_sha256(args.selected_pose_inventory),"selected_pose_content_sha256":pose_meta.get("content_sha256"),"correspondence_file_sha256":[file_sha256(p) for p in args.frozen_correspondences],"moge3_role":"token_block_3d_shape_with_one_latent_scale","arrays_sha256":arrays_sha256(arrays)}; metadata["content_sha256"]=canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output_frozen_pose_inventory,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    rows=[]
    for i,name in enumerate(names):
        with np.load(args.query_contributors/name,allow_pickle=False) as data: gt=np.asarray(data["pose_w2c"],np.float64)
        pose=output_pose[i]; te=float(np.linalg.norm(-pose[:3,:3].T@pose[:3,3]+gt[:3,:3].T@gt[:3,3])); re=float(Rotation.from_matrix(pose[:3,:3]@gt[:3,:3].T).magnitude()*180/np.pi); rows.append((te,re))
    te=np.asarray([r[0] for r in rows]); re=np.asarray([r[1] for r in rows]); report={"artifact_type":"goal_maplet_moge3_surface_registration_evaluation_v1","query_count":len(names),"accepted_count":int(accepted.sum()),"pose_frozen_before_query_pose_or_ground_truth_open":True,"median_translation_m":float(np.median(te)),"median_rotation_deg":float(np.median(re)),"p90_translation_m":float(np.quantile(te,.9)),"p90_rotation_deg":float(np.quantile(re,.9)),"threshold_hits":{f"{t:g}m_{r:g}deg":int(np.sum((te<=t)&(re<=r))) for t,r in ((2,45),(1,10),(.5,5),(.25,2),(.1,1))},"diagnostic_median_initial_surface_m":float(np.median([r.get("initial_median_m",np.nan) for r in diagnostics])),"diagnostic_median_final_surface_m":float(np.median([r.get("final_median_m",np.nan) for r in diagnostics])),"frozen_pose_file_sha256":file_sha256(args.output_frozen_pose_inventory),"frozen_pose_content_sha256":metadata["content_sha256"],"production_eligible":False}; report["content_sha256"]=canonical_json_sha256(report); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
