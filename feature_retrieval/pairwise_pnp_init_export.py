#!/usr/bin/env python3
"""Export pairwise image-matching + COLMAP 2D-3D PnP real-init poses.

This module turns a retrieval cache into a stronger real-init cache:

    query image + retrieved train image -> SIFT matches
    retrieved train keypoints -> nearest COLMAP observations -> 3D points
    query 2D + world 3D -> PnP-RANSAC query pose

The fallback for a failed PnP candidate is the original retrieved train pose,
so the exported cache remains compatible with the existing real-init evaluator.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import camera_params_to_intrinsics, read_colmap_cameras  # noqa: E402
from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)

ImageWithPoints = collections.namedtuple(
    "ImageWithPoints", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)


def read_colmap_images_with_points(path: str) -> Dict[int, ImageWithPoints]:
    """Read COLMAP ``images.bin`` including 2D observations."""
    images: Dict[int, ImageWithPoints] = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            xys = np.empty((num_points2d, 2), dtype=np.float32)
            point_ids = np.empty((num_points2d,), dtype=np.int64)
            for idx in range(num_points2d):
                x, y, point3d_id = struct.unpack("<ddq", f.read(24))
                xys[idx] = (x, y)
                point_ids[idx] = point3d_id
            images[img_id] = ImageWithPoints(img_id, qvec, tvec, camera_id, name, xys, point_ids)
    return images


def read_colmap_points3d_xyz(path: str) -> Dict[int, np.ndarray]:
    """Read COLMAP ``points3D.bin`` as ``point3D_id -> xyz``."""
    points: Dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float32)
            f.read(3)  # rgb
            f.read(8)  # reprojection error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # image_id(uint32), point2D_idx(uint32)
            points[int(point_id)] = xyz
    return points


def map_ref_keypoints_to_world_points(
    query_kpts: np.ndarray,
    ref_kpts: np.ndarray,
    colmap_xys: np.ndarray,
    colmap_point_ids: np.ndarray,
    point_xyz_by_id: Dict[int, np.ndarray],
    *,
    max_distance_px: float = 4.0,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map matched reference keypoints to nearest valid COLMAP 3D observations.

    Args:
        query_kpts: matched query keypoints in original image pixels, ``[N,2]``.
        ref_kpts: matched reference keypoints in original image pixels, ``[N,2]``.
        colmap_xys: reference image COLMAP 2D observations, ``[M,2]``.
        colmap_point_ids: COLMAP point ids aligned with ``colmap_xys``.
        point_xyz_by_id: world xyz lookup.
        max_distance_px: nearest-observation acceptance threshold.

    Returns:
        ``(pts3d_world, pts2d_query, nearest_distances)``.
    """
    query_kpts = np.asarray(query_kpts, dtype=np.float32)
    ref_kpts = np.asarray(ref_kpts, dtype=np.float32)
    colmap_xys = np.asarray(colmap_xys, dtype=np.float32)
    colmap_point_ids = np.asarray(colmap_point_ids, dtype=np.int64)
    if len(query_kpts) != len(ref_kpts):
        raise ValueError("query_kpts and ref_kpts must have the same length")
    if len(ref_kpts) == 0 or len(colmap_xys) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    pts3d: List[np.ndarray] = []
    pts2d: List[np.ndarray] = []
    distances: List[float] = []
    max_d2 = float(max_distance_px) ** 2
    for start in range(0, len(ref_kpts), int(chunk_size)):
        ref_chunk = ref_kpts[start : start + int(chunk_size)]
        d2 = ((ref_chunk[:, None, :] - colmap_xys[None, :, :]) ** 2).sum(axis=2)
        nn_idx = np.argmin(d2, axis=1)
        nn_d2 = d2[np.arange(len(ref_chunk)), nn_idx]
        for local_idx, obs_idx in enumerate(nn_idx):
            if float(nn_d2[local_idx]) > max_d2:
                continue
            point_id = int(colmap_point_ids[int(obs_idx)])
            if point_id < 0 or point_id not in point_xyz_by_id:
                continue
            global_idx = start + local_idx
            pts3d.append(np.asarray(point_xyz_by_id[point_id], dtype=np.float32))
            pts2d.append(query_kpts[global_idx].astype(np.float32))
            distances.append(float(np.sqrt(nn_d2[local_idx])))

    if not pts3d:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.stack(pts3d, axis=0).astype(np.float32),
        np.stack(pts2d, axis=0).astype(np.float32),
        np.asarray(distances, dtype=np.float32),
    )


def _camera_matrix(intrinsics: Dict[str, float]) -> np.ndarray:
    return np.array(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def solve_pnp_w2c(
    pts3d_world: np.ndarray,
    pts2d_query: np.ndarray,
    intrinsics: Dict[str, float],
    *,
    reproj_threshold: float = 8.0,
    n_iters: int = 10000,
    min_inliers: int = 12,
    use_magsac: bool = True,
) -> Tuple[np.ndarray | None, int]:
    """Solve query world-to-camera pose from 2D-3D correspondences."""
    if len(pts3d_world) < int(min_inliers):
        return None, 0
    K = _camera_matrix(intrinsics)
    pts3d = np.asarray(pts3d_world, dtype=np.float64)
    pts2d = np.asarray(pts2d_query, dtype=np.float64)
    try:
        if use_magsac and hasattr(cv2, "USAC_MAGSAC"):
            params = cv2.UsacParams()
            params.confidence = 0.999
            params.maxIterations = int(n_iters)
            params.threshold = float(reproj_threshold)
            ret = cv2.solvePnPRansac(pts3d, pts2d, K, None, params=params)
            if len(ret) == 5:
                success, _camera_matrix_out, rvec, tvec, inliers = ret
            else:
                success, rvec, tvec, inliers = ret
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d,
                pts2d,
                K,
                None,
                iterationsCount=int(n_iters),
                reprojectionError=float(reproj_threshold),
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
    except cv2.error:
        return None, 0

    if not success or inliers is None or len(inliers) < int(min_inliers):
        return None, 0 if inliers is None else int(len(inliers))

    if len(inliers) >= 6:
        try:
            ok, rvec_refined, tvec_refined = cv2.solvePnP(
                pts3d[inliers.flatten()],
                pts2d[inliers.flatten()],
                K,
                None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                rvec, tvec = rvec_refined, tvec_refined
        except cv2.error:
            pass

    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.astype(np.float32)
    pose[:3, 3] = tvec.reshape(3).astype(np.float32)
    return pose, int(len(inliers))


def build_pairwise_pnp_entries(
    *,
    query_samples: Sequence[Dict],
    candidate_results_by_query: Sequence[Sequence[Dict]],
    source_name: str,
    save_path: str | None = None,
) -> Tuple[List[Dict], Dict]:
    """Build a retrieval-init cache schema from pairwise PnP candidate results."""
    if len(query_samples) != len(candidate_results_by_query):
        raise ValueError("query_samples and candidate_results_by_query must have the same length")

    entries: List[Dict] = []
    num_success_candidates = 0
    num_queries_with_success = 0
    for sample, candidate_results in zip(query_samples, candidate_results_by_query):
        if not candidate_results:
            raise ValueError(f"query {sample.get('image_name')} has no candidates")
        ordered = sorted(
            candidate_results,
            key=lambda c: (
                1 if c.get("pnp_success") else 0,
                int(c.get("num_inliers", 0)),
                float(c.get("retrieval_score", 0.0)),
            ),
            reverse=True,
        )
        success_count = sum(1 for c in ordered if c.get("pnp_success"))
        num_success_candidates += success_count
        num_queries_with_success += int(success_count > 0)
        best = ordered[0]
        best_success = bool(best.get("pnp_success"))
        init_source = source_name if best_success else f"{source_name}_retrieval_fallback"

        candidate_poses = [np.asarray(c["pose_w2c"], dtype=np.float32) for c in ordered]
        candidate_scores = [
            float(c.get("num_inliers", 0)) if c.get("pnp_success") else float(c.get("retrieval_score", 0.0))
            for c in ordered
        ]
        entries.append(
            {
                "query_img_id": int(sample["img_id"]),
                "query_image_name": str(sample["image_name"]),
                "query_image_stem": str(sample["image_stem"]),
                "pose_init": np.asarray(best["pose_w2c"], dtype=np.float32),
                "init_source": init_source,
                "retrieval_frame_id": int(best["retrieval_frame_id"]),
                "retrieval_image_name": str(best["retrieval_image_name"]),
                "retrieval_score": float(candidate_scores[0]),
                "pose_init_candidates": np.stack(candidate_poses, axis=0).astype(np.float32),
                "candidate_valid_mask": np.ones((len(ordered),), dtype=bool),
                "retrieval_frame_ids_candidates": np.asarray(
                    [int(c["retrieval_frame_id"]) for c in ordered], dtype=np.int64
                ),
                "retrieval_image_names_candidates": np.asarray(
                    [str(c["retrieval_image_name"]) for c in ordered]
                ),
                "retrieval_scores_candidates": np.asarray(candidate_scores, dtype=np.float32),
            }
        )

    stats = {
        "method_requested": "pairwise_pnp",
        "method_used": source_name,
        "retrieval_topk_requested": int(len(candidate_results_by_query[0]) if candidate_results_by_query else 0),
        "num_query_samples": int(len(entries)),
        "num_pnp_success_candidates": int(num_success_candidates),
        "num_queries_with_pnp_success": int(num_queries_with_success),
        "query_success_rate": float(num_queries_with_success / max(1, len(entries))),
        "counts_by_source": {
            source_name: int(num_queries_with_success),
            f"{source_name}_retrieval_fallback": int(len(entries) - num_queries_with_success),
        },
    }
    if save_path:
        save_retrieval_init_entries(entries, stats, save_path)
    return entries, stats


def _load_gray_resized(path: str, resize_long_edge: int) -> Tuple[np.ndarray | None, Tuple[float, float]]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, (1.0, 1.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = gray.shape[:2]
    if resize_long_edge > 0 and max(orig_h, orig_w) > resize_long_edge:
        scale = float(resize_long_edge) / float(max(orig_h, orig_w))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_h, new_w = orig_h, orig_w
    return gray, (float(orig_w) / float(new_w), float(orig_h) / float(new_h))


def match_sift_keypoints(
    query_img_path: str,
    ref_img_path: str,
    *,
    resize_long_edge: int = 1280,
    max_features: int = 8192,
    ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return matched ``(query_xy_orig, ref_xy_orig, info)`` using SIFT."""
    query_gray, query_scale = _load_gray_resized(query_img_path, resize_long_edge)
    ref_gray, ref_scale = _load_gray_resized(ref_img_path, resize_long_edge)
    info = {
        "num_query_keypoints": 0,
        "num_ref_keypoints": 0,
        "num_raw_matches": 0,
        "num_ratio_matches": 0,
    }
    if query_gray is None or ref_gray is None:
        info["failure_reason"] = "image_read_failed"
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), info
    sift = cv2.SIFT_create(nfeatures=int(max_features))
    q_kp, q_desc = sift.detectAndCompute(query_gray, None)
    r_kp, r_desc = sift.detectAndCompute(ref_gray, None)
    info["num_query_keypoints"] = len(q_kp)
    info["num_ref_keypoints"] = len(r_kp)
    if q_desc is None or r_desc is None or len(q_kp) == 0 or len(r_kp) == 0:
        info["failure_reason"] = "no_sift_descriptors"
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), info
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(q_desc, r_desc, k=2)
    info["num_raw_matches"] = len(raw)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < float(ratio) * n.distance:
            good.append(m)
    info["num_ratio_matches"] = len(good)
    if not good:
        info["failure_reason"] = "no_ratio_matches"
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), info
    query_pts = np.array([q_kp[m.queryIdx].pt for m in good], dtype=np.float32)
    ref_pts = np.array([r_kp[m.trainIdx].pt for m in good], dtype=np.float32)
    query_pts[:, 0] *= query_scale[0]
    query_pts[:, 1] *= query_scale[1]
    ref_pts[:, 0] *= ref_scale[0]
    ref_pts[:, 1] *= ref_scale[1]
    return query_pts.astype(np.float32), ref_pts.astype(np.float32), info


def _resolve_image_path(images_root: str, image_name: str) -> str:
    return os.path.join(images_root, image_name)


def export_pairwise_pnp_init(
    *,
    retrieval_init_path: str,
    colmap_dir: str,
    query_split: str,
    images_root: str,
    save_path: str,
    topk: int = 10,
    source_name: str = "pairwise_sift_pnp",
    resize_long_edge: int = 1280,
    max_features: int = 8192,
    ratio: float = 0.75,
    max_observation_distance_px: float = 5.0,
    min_correspondences: int = 24,
    min_inliers: int = 12,
    reproj_threshold: float = 8.0,
    pnp_iters: int = 10000,
    max_queries: int = 0,
    use_magsac: bool = True,
) -> Dict:
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    if max_queries and int(max_queries) > 0:
        query_samples = query_samples[: int(max_queries)]
    query_by_name = {sample["image_name"]: sample for sample in query_samples}
    entries, retrieval_stats = load_retrieval_init_entries(retrieval_init_path)
    entries = [entry for entry in entries if entry["query_image_name"] in query_by_name]
    entries = entries[: len(query_samples)] if max_queries and int(max_queries) > 0 else entries

    cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    intrinsics = camera_params_to_intrinsics(first_cam)
    images_with_points = read_colmap_images_with_points(os.path.join(colmap_dir, "images.bin"))
    image_points_by_id = {int(img_id): meta for img_id, meta in images_with_points.items()}
    points3d = read_colmap_points3d_xyz(os.path.join(colmap_dir, "points3D.bin"))

    candidate_results_by_query: List[List[Dict]] = []
    used_query_samples: List[Dict] = []
    match_counts: List[int] = []
    corr_counts: List[int] = []
    inlier_counts: List[int] = []
    failure_counts: Dict[str, int] = {}

    for qi, entry in enumerate(entries):
        query_sample = query_by_name[entry["query_image_name"]]
        used_query_samples.append(query_sample)
        query_path = _resolve_image_path(images_root, entry["query_image_name"])
        candidate_results: List[Dict] = []
        candidate_poses = np.asarray(entry["pose_init_candidates"], dtype=np.float32)
        candidate_frame_ids = np.asarray(entry["retrieval_frame_ids_candidates"], dtype=np.int64)
        candidate_names = list(entry["retrieval_image_names_candidates"])
        candidate_scores = np.asarray(entry["retrieval_scores_candidates"], dtype=np.float32)
        valid_mask = np.asarray(entry["candidate_valid_mask"], dtype=bool)
        valid_indices = [idx for idx, keep in enumerate(valid_mask[: int(topk)]) if keep]
        for cand_idx in valid_indices:
            ref_img_id = int(candidate_frame_ids[cand_idx])
            ref_name = str(candidate_names[cand_idx])
            fallback_pose = candidate_poses[cand_idx]
            ref_meta = image_points_by_id.get(ref_img_id)
            ref_path = _resolve_image_path(images_root, ref_name)
            result = {
                "pose_w2c": fallback_pose,
                "retrieval_frame_id": ref_img_id,
                "retrieval_image_name": ref_name,
                "retrieval_score": float(candidate_scores[cand_idx]),
                "pnp_success": False,
                "num_matches": 0,
                "num_correspondences": 0,
                "num_inliers": 0,
            }
            if ref_meta is None:
                result["failure_reason"] = "missing_ref_colmap_observations"
                failure_counts[result["failure_reason"]] = failure_counts.get(result["failure_reason"], 0) + 1
                candidate_results.append(result)
                continue
            query_kpts, ref_kpts, match_info = match_sift_keypoints(
                query_path,
                ref_path,
                resize_long_edge=resize_long_edge,
                max_features=max_features,
                ratio=ratio,
            )
            result["num_matches"] = int(len(query_kpts))
            match_counts.append(int(len(query_kpts)))
            if len(query_kpts) < int(min_correspondences):
                reason = match_info.get("failure_reason", "too_few_sift_matches")
                result["failure_reason"] = reason
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
                candidate_results.append(result)
                continue
            pts3d, pts2d, _distances = map_ref_keypoints_to_world_points(
                query_kpts,
                ref_kpts,
                ref_meta.xys,
                ref_meta.point3D_ids,
                points3d,
                max_distance_px=max_observation_distance_px,
            )
            result["num_correspondences"] = int(len(pts3d))
            corr_counts.append(int(len(pts3d)))
            pose, num_inliers = solve_pnp_w2c(
                pts3d,
                pts2d,
                intrinsics,
                reproj_threshold=reproj_threshold,
                n_iters=pnp_iters,
                min_inliers=min_inliers,
                use_magsac=use_magsac,
            )
            result["num_inliers"] = int(num_inliers)
            inlier_counts.append(int(num_inliers))
            if pose is None:
                result["failure_reason"] = "pnp_failed"
                failure_counts["pnp_failed"] = failure_counts.get("pnp_failed", 0) + 1
            else:
                result["pose_w2c"] = pose.astype(np.float32)
                result["pnp_success"] = True
                result["failure_reason"] = ""
            candidate_results.append(result)
        if not candidate_results:
            candidate_results.append(
                {
                    "pose_w2c": np.asarray(entry["pose_init"], dtype=np.float32),
                    "retrieval_frame_id": int(entry["retrieval_frame_id"]),
                    "retrieval_image_name": str(entry["retrieval_image_name"]),
                    "retrieval_score": float(entry["retrieval_score"]),
                    "pnp_success": False,
                    "num_inliers": 0,
                    "failure_reason": "no_valid_retrieval_candidates",
                }
            )
        candidate_results_by_query.append(candidate_results)
        if (qi + 1) % 10 == 0 or qi == 0:
            successes = sum(1 for group in candidate_results_by_query for c in group if c.get("pnp_success"))
            print(f"[{qi + 1}/{len(entries)}] pairwise PnP success candidates: {successes}")

    output_entries, stats = build_pairwise_pnp_entries(
        query_samples=used_query_samples,
        candidate_results_by_query=candidate_results_by_query,
        source_name=source_name,
        save_path=save_path,
    )
    stats.update(
        {
            "retrieval_init_path": str(retrieval_init_path),
            "retrieval_stats": retrieval_stats,
            "images_root": str(images_root),
            "resize_long_edge": int(resize_long_edge),
            "max_features": int(max_features),
            "ratio": float(ratio),
            "max_observation_distance_px": float(max_observation_distance_px),
            "min_correspondences": int(min_correspondences),
            "min_inliers": int(min_inliers),
            "reproj_threshold": float(reproj_threshold),
            "failure_counts": failure_counts,
            "mean_sift_matches": float(np.mean(match_counts)) if match_counts else 0.0,
            "mean_2d3d_correspondences": float(np.mean(corr_counts)) if corr_counts else 0.0,
            "mean_pnp_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
            "num_exported_entries": int(len(output_entries)),
        }
    )
    # save updated stats with the same entries
    save_retrieval_init_entries(output_entries, stats, save_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pairwise SIFT+COLMAP-PnP real-init cache")
    parser.add_argument("--retrieval_init_path", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--source_name", default="pairwise_sift_pnp")
    parser.add_argument("--resize_long_edge", type=int, default=1280)
    parser.add_argument("--max_features", type=int, default=8192)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--max_observation_distance_px", type=float, default=5.0)
    parser.add_argument("--min_correspondences", type=int, default=24)
    parser.add_argument("--min_inliers", type=int, default=12)
    parser.add_argument("--reproj_threshold", type=float, default=8.0)
    parser.add_argument("--pnp_iters", type=int, default=10000)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--no_magsac", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_pairwise_pnp_init(
        retrieval_init_path=args.retrieval_init_path,
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        images_root=args.images_root,
        save_path=args.save_path,
        topk=args.topk,
        source_name=args.source_name,
        resize_long_edge=args.resize_long_edge,
        max_features=args.max_features,
        ratio=args.ratio,
        max_observation_distance_px=args.max_observation_distance_px,
        min_correspondences=args.min_correspondences,
        min_inliers=args.min_inliers,
        reproj_threshold=args.reproj_threshold,
        pnp_iters=args.pnp_iters,
        max_queries=args.max_queries,
        use_magsac=not args.no_magsac,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
