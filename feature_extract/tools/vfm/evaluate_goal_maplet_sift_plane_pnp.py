"""PlanarReloc-style masked high-resolution SIFT matching and metric PnP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _camera_inventory,
    _radio,
    _records,
    _region_tokens,
    _scaled_intrinsics,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import RenderedPlaneObservations
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import inverse_simple_radial


def _grid_indices(points_xy: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(points_xy, np.float64).reshape(-1, 2)
    x = np.rint((point[:, 0] + 0.5) * 256.0 / float(width) - 0.5).astype(np.int64)
    y = np.rint((point[:, 1] + 0.5) * 144.0 / float(height) - 0.5).astype(np.int64)
    return np.clip(x, 0, 255), np.clip(y, 0, 143)


def _mutual_ratio_matches(
    query: np.ndarray, source: np.ndarray, *, ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(query) < 2 or len(source) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(np.asarray(query, np.float32), np.asarray(source, np.float32), k=2)
    reverse = matcher.knnMatch(np.asarray(source, np.float32), np.asarray(query, np.float32), k=2)
    reverse_best = {
        int(row[0].queryIdx): int(row[0].trainIdx)
        for row in reverse if len(row) == 2 and row[0].distance < float(ratio) * row[1].distance
    }
    qi, si, distance = [], [], []
    for row in forward:
        if len(row) != 2 or row[0].distance >= float(ratio) * row[1].distance:
            continue
        match = row[0]
        if reverse_best.get(int(match.trainIdx)) != int(match.queryIdx):
            continue
        qi.append(int(match.queryIdx)); si.append(int(match.trainIdx)); distance.append(float(match.distance))
    return np.asarray(qi, np.int64), np.asarray(si, np.int64), np.asarray(distance, np.float32)


def _homography_inliers(query_xy: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    if len(query_xy) < 6:
        return np.zeros(len(query_xy), bool)
    cv2.setRNGSeed(260901)
    _, mask = cv2.findHomography(
        np.asarray(query_xy, np.float64), np.asarray(source_xy, np.float64),
        cv2.RANSAC, 4.0, maxIters=2000, confidence=0.995,
    )
    return np.zeros(len(query_xy), bool) if mask is None else mask.reshape(-1).astype(bool)


def _mapping_world_point(
    point_xy: np.ndarray,
    labels: np.ndarray,
    local_plane: int,
    contributor: Path,
    contributor_cache: dict[Path, tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray]] | None = None,
) -> np.ndarray | None:
    cached = None if contributor_cache is None else contributor_cache.get(contributor)
    if cached is None:
        with np.load(contributor, allow_pickle=False) as data:
            cached = (
                np.asarray(data["dominant_depth"], np.float64),
                np.asarray(data["pose_w2c"], np.float64),
                int(data["camera_model_id"]),
                int(data["camera_width"]),
                int(data["camera_height"]),
                np.asarray(data["camera_params"], np.float64),
            )
        if contributor_cache is not None:
            contributor_cache[contributor] = cached
    depth, pose, model, width, height, params = cached
    gx, gy = _grid_indices(np.asarray(point_xy).reshape(1, 2), width, height)
    x0, y0 = int(gx[0]), int(gy[0])
    yy, xx = np.mgrid[max(0, y0 - 1):min(144, y0 + 2), max(0, x0 - 1):min(256, x0 + 2)]
    keep = (labels[yy, xx] == int(local_plane)) & np.isfinite(depth[yy, xx]) & (depth[yy, xx] > 0)
    if not np.any(keep):
        return None
    xx = xx[keep].astype(np.float64); yy = yy[keep].astype(np.float64); z = depth[yy.astype(int), xx.astype(int)]
    K, k1 = _scaled_intrinsics(model, params, width, height)
    distorted = np.c_[(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1]]
    ideal = inverse_simple_radial(distorted, k1)
    camera = np.c_[ideal * z[:, None], z]
    rotation, translation = pose[:3, :3], pose[:3, 3]
    center = -rotation.T @ translation
    return np.median(camera @ rotation + center, axis=0)


def _solve(world: np.ndarray, pixels: np.ndarray, K: np.ndarray, k1: float, rows: np.ndarray) -> np.ndarray | None:
    if len(rows) < 6:
        return None
    distortion = np.asarray([k1, 0, 0, 0, 0], np.float64)
    cv2.setRNGSeed(260901)
    ok, rvec, tvec, inlier = cv2.solvePnPRansac(
        world[rows], pixels[rows], K, distortion, iterationsCount=2000,
        reprojectionError=4.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inlier is None or len(inlier) < 6:
        return None
    selected = rows[inlier.reshape(-1)]
    rvec, tvec = cv2.solvePnPRefineLM(world[selected], pixels[selected], K, distortion, rvec, tvec)
    pose = np.eye(4); pose[:3, :3] = cv2.Rodrigues(rvec)[0]; pose[:3, 3] = np.asarray(tvec).reshape(3)
    return pose


def _score(
    pose: np.ndarray, world: np.ndarray, pixels: np.ndarray, query_keypoint: np.ndarray,
    K: np.ndarray, k1: float,
) -> tuple[int, float]:
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], K,
        np.asarray([k1, 0, 0, 0, 0], np.float64),
    )
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    valid = (camera[:, 2] > 0) & (residual <= 4.0)
    chosen = []
    for keypoint in np.unique(query_keypoint):
        rows = np.flatnonzero(valid & (query_keypoint == keypoint))
        if len(rows):
            chosen.append(int(rows[np.argmin(residual[rows])]))
    return len(chosen), float(np.median(residual[chosen])) if chosen else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--observation_dir", type=Path, required=True)
    parser.add_argument("--observation_field", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--plane_ranking", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--query_camera_inventory", type=Path, required=True)
    parser.add_argument("--mapping_contributors", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--query_indices", type=str, default="")
    parser.add_argument("--topk_planes", type=int, default=10)
    parser.add_argument("--top_source_views", type=int, default=4)
    parser.add_argument("--output_candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists() or args.output_candidates.exists():
        raise FileExistsError("refusing to overwrite masked SIFT plane PnP diagnostic")

    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    with np.load(args.observation_field, allow_pickle=False) as data:
        field_meta = json.loads(str(data["metadata_json"].item()))
        field = np.asarray(data["observation_descriptors"], np.float32)
    if field_meta.get("visibility_atlas_content_sha256") != atlas_meta.get("content_sha256"):
        raise ValueError("observation descriptor field differs from visibility atlas")
    paths = sorted(args.observation_dir.glob("*.npz"))
    global_lookup: list[tuple[Path, int, str]] = []
    for path in paths:
        observation, metadata = RenderedPlaneObservations.load_npz(path)
        for local in range(len(observation.normals_world)):
            global_lookup.append((path, local, str(metadata["source_name"])))
    if np.max(atlas.plane_observation_rows) >= len(global_lookup):
        raise ValueError("visibility lineage exceeds observation inventory")
    ranking = json.loads(args.plane_ranking.read_text())
    rows = ranking["rows"]
    if args.query_indices:
        indices = [int(value) for value in args.query_indices.split(",")]
        rows = [rows[index] for index in indices]
    radio = _records(args.radio_manifest)
    cameras, camera_meta = _camera_inventory(args.query_camera_inventory)
    sift = cv2.SIFT_create(nfeatures=4096)
    image_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    observation_cache: dict[Path, np.ndarray] = {}
    contributor_cache: dict[Path, tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray]] = {}
    mapping_point_cache: dict[tuple[str, int, int], np.ndarray | None] = {}

    def features(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if name not in image_cache:
            path = args.image_root / (name[:-4] if name.endswith(".npz") else name)
            if not path.exists() and "__" in name:
                nested = (name[:-4] if name.endswith(".npz") else name).replace("__", "/", 1)
                path = args.image_root / nested
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            if name in cameras:
                _, target_width, target_height, _ = cameras[name]
            else:
                with np.load(args.mapping_contributors / name, allow_pickle=False) as data:
                    target_width = int(data["camera_width"])
                    target_height = int(data["camera_height"])
            if image.shape != (target_height, target_width):
                image = cv2.resize(
                    image, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            keypoint, descriptor = sift.detectAndCompute(image, None)
            xy = np.asarray([point.pt for point in keypoint], np.float64).reshape(-1, 2)
            descriptor = np.zeros((0, 128), np.float32) if descriptor is None else descriptor
            image_cache[name] = xy, descriptor, np.asarray(image.shape[:2], np.int64)
        return image_cache[name]

    all_names, all_candidates, all_diagnostics = [], [], []
    for query_index, query in enumerate(rows):
        name = str(query["image"]); all_names.append(name)
        qxy, qdescriptor, qshape = features(name)
        qx, qy = _grid_indices(qxy, int(qshape[1]), int(qshape[0]))
        query_planes, _ = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        query_radio = _radio(name, radio)
        world_rows, pixel_rows, key_rows, plane_rows, distance_rows = [], [], [], [], []
        for region in query["regions"]:
            region_id = int(region["region"])
            qkp = np.flatnonzero(query_planes.labels[qy, qx] == region_id)
            if len(qkp) < 6:
                continue
            token = _region_tokens(query_planes.labels, region_id)
            if not len(token):
                continue
            qmean = np.mean(query_radio[token], axis=0); qmean /= max(float(np.linalg.norm(qmean)), 1e-8)
            for plane_rank, plane in enumerate(region["top10"][: int(args.topk_planes)]):
                plane = int(plane); lo, hi = map(int, atlas.plane_offsets[plane:plane + 2])
                view_order = np.argsort(-(field[lo:hi] @ qmean), kind="stable")[: int(args.top_source_views)]
                for local_view in view_order.tolist():
                    atlas_row = lo + local_view
                    global_row = int(atlas.plane_observation_rows[atlas_row])
                    observation_path, local_plane, source_name = global_lookup[global_row]
                    if observation_path not in observation_cache:
                        observation_cache[observation_path] = RenderedPlaneObservations.load_npz(observation_path)[0].labels
                    source_labels = observation_cache[observation_path]
                    sxy, sdescriptor, sshape = features(source_name)
                    sx, sy = _grid_indices(sxy, int(sshape[1]), int(sshape[0]))
                    skp = np.flatnonzero(source_labels[sy, sx] == local_plane)
                    if len(skp) < 6:
                        continue
                    qi, si, distance = _mutual_ratio_matches(qdescriptor[qkp], sdescriptor[skp])
                    if len(qi) < 6:
                        continue
                    keep = _homography_inliers(qxy[qkp[qi]], sxy[skp[si]])
                    for match in np.flatnonzero(keep).tolist():
                        source_index = int(skp[si[match]])
                        cache_key = (source_name, source_index, int(local_plane))
                        if cache_key not in mapping_point_cache:
                            mapping_point_cache[cache_key] = _mapping_world_point(
                                sxy[source_index], source_labels, local_plane,
                                args.mapping_contributors / source_name,
                                contributor_cache,
                            )
                        point = mapping_point_cache[cache_key]
                        if point is None:
                            continue
                        world_rows.append(point); pixel_rows.append(qxy[int(qkp[qi[match]])])
                        key_rows.append(int(qkp[qi[match]])); plane_rows.append(plane)
                        distance_rows.append(float(distance[match]) + 5.0 * plane_rank)
        if world_rows:
            world = np.asarray(world_rows, np.float64); pixel = np.asarray(pixel_rows, np.float64)
            key = np.asarray(key_rows, np.int64); plane_id = np.asarray(plane_rows, np.int64)
            distance = np.asarray(distance_rows, np.float64)
            chosen = []
            for keypoint in np.unique(key):
                order = np.flatnonzero(key == keypoint)
                order = order[np.argsort(distance[order], kind="stable")]
                local = []
                for row in order.tolist():
                    if any(np.linalg.norm(world[row] - world[prior]) <= 0.5 for prior in local):
                        continue
                    local.append(row)
                    if len(local) == 3:
                        break
                chosen.extend(local)
            chosen = np.asarray(chosen, np.int64)
            world, pixel, key, plane_id = world[chosen], pixel[chosen], key[chosen], plane_id[chosen]
        else:
            world = np.zeros((0, 3)); pixel = np.zeros((0, 2)); key = np.zeros(0, int); plane_id = np.zeros(0, int)
        model, width, height, params = cameras[name]
        K, k1 = _scaled_intrinsics(model, params, width, height, width=int(qshape[1]), height=int(qshape[0]))
        groups = [("all", np.arange(len(world), dtype=np.int64))]
        for plane in np.unique(plane_id):
            selected = np.flatnonzero(plane_id == plane)
            if len(selected) >= 6:
                groups.append(("plane", selected))
        candidates = []
        for origin, selected in groups:
            pose = _solve(world, pixel, K, k1, selected)
            if pose is None:
                continue
            inliers, median = _score(pose, world, pixel, key, K, k1)
            candidates.append({"pose": pose, "origin": origin, "inliers": inliers, "median": median})
        candidates.sort(key=lambda row: (-row["inliers"], row["median"], row["origin"]))
        all_candidates.append(candidates)
        all_diagnostics.append({
            "name": name, "sift_keypoint_count": int(len(qxy)),
            "correspondence_count": int(len(world)), "unique_query_keypoint_count": int(len(np.unique(key))),
            "candidate_count": int(len(candidates)),
        })
        print(
            f"{query_index + 1}/{len(rows)} {name}: "
            f"{len(world)} correspondences, {len(candidates)} candidates",
            flush=True,
        )

    offsets = np.r_[0, np.cumsum([len(row) for row in all_candidates])].astype(np.int64)
    arrays = {
        "names": np.asarray(all_names), "candidate_offsets": offsets,
        "candidate_pose_w2c": np.asarray([row["pose"] for rows in all_candidates for row in rows], np.float64).reshape(-1, 4, 4),
        "candidate_inlier_count": np.asarray([row["inliers"] for rows in all_candidates for row in rows], np.int64),
        "candidate_origin": np.asarray([row["origin"] for rows in all_candidates for row in rows]),
    }
    metadata = {
        "artifact_type": "goal_maplet_masked_sift_plane_pnp_candidate_inventory_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(all_names)),
        "pose_or_ground_truth_opened": False, "query_depth_used": False,
        "local_matcher": "OpenCV_SIFT_mutual_Lowe0.8_plus_finite_plane_homography_RANSAC",
        "radio_role": "finite_plane_and_source_view_retrieval_only",
        "rgb_resize": "area_resize_to_each_frozen_camera_contract_before_SIFT",
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "plane_ranking_file_sha256": file_sha256(args.plane_ranking),
        "query_camera_inventory_content_sha256": camera_meta.get("content_sha256"),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_candidates, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))

    postlabel, success, oracle = [], [], []
    for index, name in enumerate(all_names):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        gt_center = -gt[:3, :3].T @ gt[:3, 3]
        errors = []
        for candidate in all_candidates[index]:
            pose = candidate["pose"]; center = -pose[:3, :3].T @ pose[:3, 3]
            errors.append((
                float(np.linalg.norm(center - gt_center)),
                float(Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude() * 180 / np.pi),
            ))
        selected = errors[0] if errors else (float("inf"), float("inf"))
        row_oracle = bool(any(t <= 2 and r <= 45 for t, r in errors))
        success.append(selected[0] <= 2 and selected[1] <= 45); oracle.append(row_oracle)
        postlabel.append({"name": name, "translation_error_m": selected[0], "rotation_error_deg": selected[1], "candidate_oracle_2m45": row_oracle, "minimum_candidate_translation_m": None if not errors else min(t for t, _ in errors)})
    report = {
        "artifact_type": "goal_maplet_masked_sift_plane_pnp_development_evaluation_v1",
        "candidate_inventory_file_sha256": file_sha256(args.output_candidates),
        "candidate_inventory_content_sha256": metadata["content_sha256"],
        "selection_frozen_before_query_pose_opened": True,
        "query_count": int(len(all_names)), "selected_recall_2m45": float(np.mean(success)),
        "candidate_oracle_2m45": float(np.mean(oracle)), "pose_free_rows": all_diagnostics,
        "postlabel_rows": postlabel, "elapsed_seconds": float(time.perf_counter() - started),
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("pose_free_rows", "postlabel_rows")}, indent=2))


if __name__ == "__main__":
    main()
