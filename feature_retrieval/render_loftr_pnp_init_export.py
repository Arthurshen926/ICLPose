#!/usr/bin/env python3
"""Export NetVLAD/retrieval-guided render-depth LoFTR+PnP init poses.

Unlike oracle LoFTR diagnostics, this entrypoint consumes a real retrieval
cache. For every query and retrieved reference candidate it renders 2DGS depth
at the reference pose, matches query/reference RGB with LoFTR, unprojects
reference matches using rendered depth, solves PnP, and writes the standard
real-init cache consumed by the existing full-pipeline evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import camera_params_to_intrinsics, read_colmap_cameras  # noqa: E402
from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)
from feature_retrieval.pairwise_pnp_init_export import build_pairwise_pnp_entries  # noqa: E402
from pose_refine.evaluate_pipeline import _render_depth_for_loftr  # noqa: E402
from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution, _scale_intrinsics  # noqa: E402


def candidate_ref_entries_from_retrieval_entry(entry: Dict, *, topk: int) -> List[Dict]:
    """Extract valid topK reference entries from a retrieval-init schema entry."""
    poses = np.asarray(entry["pose_init_candidates"], dtype=np.float32)
    valid = np.asarray(entry["candidate_valid_mask"], dtype=bool)
    frame_ids = np.asarray(entry["retrieval_frame_ids_candidates"], dtype=np.int64)
    names = list(entry["retrieval_image_names_candidates"])
    scores = np.asarray(entry["retrieval_scores_candidates"], dtype=np.float32)
    refs: List[Dict] = []
    for idx in range(min(int(topk), len(poses))):
        if not bool(valid[idx]):
            continue
        refs.append(
            {
                "retrieval_frame_id": int(frame_ids[idx]),
                "retrieval_image_name": str(names[idx]),
                "retrieval_score": float(scores[idx]),
                "fallback_pose_w2c": poses[idx].astype(np.float32),
            }
        )
    return refs


def slice_retrieval_entries(entries: Sequence[Dict], *, query_start: int = 0, max_queries: int = 0) -> List[Dict]:
    start = max(0, int(query_start))
    sliced = list(entries)[start:]
    if max_queries and int(max_queries) > 0:
        sliced = sliced[: int(max_queries)]
    return sliced


def loftr_result_to_candidate(
    result,
    *,
    fallback_pose_w2c: np.ndarray,
    retrieval_frame_id: int,
    retrieval_image_name: str,
    retrieval_score: float,
) -> Dict:
    """Convert a LoFTRResult-like object to a cache candidate dict."""
    success = bool(getattr(result, "success", False)) and getattr(result, "pose_w2c", None) is not None
    pose = np.asarray(result.pose_w2c if success else fallback_pose_w2c, dtype=np.float32)
    return {
        "pose_w2c": pose,
        "retrieval_frame_id": int(retrieval_frame_id),
        "retrieval_image_name": str(retrieval_image_name),
        "retrieval_score": float(retrieval_score),
        "pnp_success": success,
        "num_inliers": int(getattr(result, "num_inliers", 0)),
        "num_matches": int(getattr(result, "num_confident_matches", 0)),
        **(getattr(result, "extra", {}) or {}).get("pnp_quality", {}),
        "failure_reason": "" if success else str(getattr(result, "failure_reason", "")),
    }


def attach_pnp_quality_stats(result, pose_w2c: np.ndarray, intrinsics_loftr: Dict[str, float]) -> None:
    if not bool(getattr(result, "success", False)):
        return
    extra = getattr(result, "extra", {}) or {}
    pts3d = extra.get("pts3d_world")
    query_xy = extra.get("query_keypoints")
    if pts3d is None or query_xy is None:
        return
    pts3d = np.asarray(pts3d, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if pts3d.size == 0 or query_xy.size == 0:
        return
    proj_xy, valid = _project_world_points(pts3d, np.asarray(pose_w2c, dtype=np.float64), intrinsics_loftr)
    count = min(len(proj_xy), len(query_xy), len(valid))
    if count == 0:
        return
    errors = np.linalg.norm(proj_xy[:count] - query_xy[:count], axis=1)
    valid = np.asarray(valid[:count], dtype=bool)
    inlier_mask = np.asarray(extra.get("pnp_inlier_mask", valid), dtype=bool).reshape(-1)[:count] & valid
    valid_errors = errors[valid]
    inlier_errors = errors[inlier_mask]
    if inlier_errors.size == 0:
        inlier_errors = valid_errors
    if inlier_errors.size == 0:
        return
    confidence = np.asarray(extra.get("confidence", []), dtype=np.float64).reshape(-1)[:count]
    inlier_conf = confidence[inlier_mask] if confidence.size >= count and inlier_mask.any() else confidence
    conf_mean = float(np.mean(inlier_conf)) if inlier_conf.size else float(getattr(result, "mean_confidence", 0.0))
    num_inliers = float(getattr(result, "num_inliers", int(inlier_mask.sum())))
    num_matches = float(max(1, int(getattr(result, "num_confident_matches", count))))
    rmse = float(np.sqrt(np.mean(np.square(inlier_errors))))
    inlier_ratio = float(num_inliers / num_matches)
    selection_score = float((num_inliers * max(inlier_ratio, 1e-6) * max(conf_mean, 1e-6)) / (1.0 + rmse))
    extra["pnp_quality"] = {
        "pnp_reproj_rmse": rmse,
        "pnp_reproj_median": float(np.median(inlier_errors)),
        "pnp_inlier_ratio": inlier_ratio,
        "pnp_inlier_conf_mean": conf_mean,
        "selection_score": selection_score,
    }
    result.extra = extra


def _sample_image_path(images_root: str, image_name: str) -> str:
    return os.path.join(images_root, image_name)


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    return (-R.T @ t).astype(np.float32)


def _pose_error(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[float, float]:
    R_rel = pred_w2c[:3, :3].T @ gt_w2c[:3, :3]
    trace = float(np.clip(np.trace(R_rel), -1.0, 3.0))
    angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    trans = float(np.linalg.norm(_camera_center_from_w2c(pred_w2c) - _camera_center_from_w2c(gt_w2c)))
    return float(angle), trans


def _feature_stem(image_name: str) -> str:
    return Path(str(image_name).replace("\\", "/")).with_suffix("").as_posix().replace("/", "_")


def _project_world_points(points_world: np.ndarray, pose_w2c: np.ndarray, intrinsics: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64)
    pts_cam = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > 1e-6)
    xy = np.zeros((len(points), 2), dtype=np.float32)
    xy[:, 0] = (intrinsics["fx"] * pts_cam[:, 0] / np.maximum(z, 1e-6) + intrinsics["cx"]).astype(np.float32)
    xy[:, 1] = (intrinsics["fy"] * pts_cam[:, 1] / np.maximum(z, 1e-6) + intrinsics["cy"]).astype(np.float32)
    valid = valid & np.isfinite(xy).all(axis=1)
    return xy, valid


def _teacher_correspondence_payload(
    result,
    *,
    query_pose_w2c: np.ndarray,
    intrinsics_loftr: Dict[str, float],
    loftr_hw: tuple[int, int],
    max_points: int,
    inliers_only: bool,
) -> Dict | None:
    extra = getattr(result, "extra", {}) or {}
    required = ["query_keypoints", "pts3d_world", "confidence"]
    if not all(key in extra for key in required):
        return None
    query_xy = np.asarray(extra["query_keypoints"], dtype=np.float32)
    pts3d_world = np.asarray(extra["pts3d_world"], dtype=np.float32)
    conf = np.asarray(extra["confidence"], dtype=np.float32).reshape(-1)
    if len(query_xy) == 0 or len(pts3d_world) == 0:
        return None
    n = min(len(query_xy), len(pts3d_world), len(conf))
    query_xy = query_xy[:n]
    pts3d_world = pts3d_world[:n]
    conf = conf[:n]
    valid = np.ones((n,), dtype=bool)
    if inliers_only and "pnp_inlier_mask" in extra:
        valid &= np.asarray(extra["pnp_inlier_mask"], dtype=bool).reshape(-1)[:n]
    map_xy, proj_valid = _project_world_points(pts3d_world, query_pose_w2c, intrinsics_loftr)
    valid &= proj_valid
    h, w = int(loftr_hw[0]), int(loftr_hw[1])
    valid &= (
        (query_xy[:, 0] >= 0.0)
        & (query_xy[:, 0] <= w - 1)
        & (query_xy[:, 1] >= 0.0)
        & (query_xy[:, 1] <= h - 1)
        & (map_xy[:, 0] >= 0.0)
        & (map_xy[:, 0] <= w - 1)
        & (map_xy[:, 1] >= 0.0)
        & (map_xy[:, 1] <= h - 1)
    )
    if not valid.any():
        return None
    query_xy = query_xy[valid]
    map_xy = map_xy[valid]
    pts3d_world = pts3d_world[valid]
    conf = conf[valid]
    order = np.argsort(-conf)[: int(max_points)]
    return {
        "query_xy": query_xy[order].astype(np.float32),
        "map_xy": map_xy[order].astype(np.float32),
        "pts3d_world": pts3d_world[order].astype(np.float32),
        "confidence": conf[order].astype(np.float32),
        "query_hw": np.asarray(loftr_hw, dtype=np.int32),
        "map_hw": np.asarray(loftr_hw, dtype=np.int32),
        "coordinate_space": np.asarray("image"),
        "source": np.asarray("netvlad_render_loftr_pnp_teacher"),
    }


def export_render_loftr_pnp_init(
    *,
    retrieval_init_path: str,
    colmap_dir: str,
    query_split: str,
    images_root: str,
    ply_path: str,
    save_path: str,
    topk: int = 10,
    source_name: str = "netvlad_render_loftr_pnp",
    gpu: int = 0,
    loftr_long_edge: int = 840,
    loftr_conf: float = 0.3,
    reproj_threshold: float = 8.0,
    pnp_iters: int = 10000,
    render_long_edge: int = 840,
    query_start: int = 0,
    max_queries: int = 0,
    use_magsac: bool = False,
    save_correspondence_dir: str | None = None,
    correspondence_max_points: int = 512,
    correspondence_inliers_only: bool = True,
) -> Dict:
    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    query_samples_all = list_colmap_split_samples(colmap_dir, query_split)
    query_by_name = {sample["image_name"]: sample for sample in query_samples_all}
    retrieval_entries, retrieval_stats = load_retrieval_init_entries(retrieval_init_path)
    retrieval_entries = [entry for entry in retrieval_entries if entry["query_image_name"] in query_by_name]
    retrieval_entries = slice_retrieval_entries(
        retrieval_entries,
        query_start=int(query_start),
        max_queries=int(max_queries),
    )

    cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    intrinsics = camera_params_to_intrinsics(first_cam)
    orig_hw = (int(first_cam.height), int(first_cam.width))
    scale = float(render_long_edge) / float(max(orig_hw))
    render_hw = (
        max(8, int(round(orig_hw[0] * scale / 8.0)) * 8),
        max(8, int(round(orig_hw[1] * scale / 8.0)) * 8),
    )
    loftr_hw = _compute_loftr_resolution(orig_hw, loftr_long_edge)
    intr_loftr = _scale_intrinsics(intrinsics, orig_hw, loftr_hw)
    corr_dir = Path(save_correspondence_dir) if save_correspondence_dir else None
    if corr_dir is not None:
        corr_dir.mkdir(parents=True, exist_ok=True)

    from feature_gaussian import HybridGaussianModel

    gaussians_depth = HybridGaussianModel(sh_degree=3, latent_dim=0)
    gaussians_depth.load_ply(ply_path, freeze_geometry=True)
    gaussians_depth.active_sh_degree = 3
    gaussians_depth = gaussians_depth.to(device) if hasattr(gaussians_depth, "to") else gaussians_depth

    loftr = LoFTRInitializer(
        device=device,
        loftr_long_edge=loftr_long_edge,
        confidence_threshold=loftr_conf,
        reproj_threshold=reproj_threshold,
        pnp_iters=pnp_iters,
        use_magsac=use_magsac,
    )

    used_query_samples: List[Dict] = []
    candidate_results_by_query: List[List[Dict]] = []
    failure_counts: Dict[str, int] = {}
    inlier_counts: List[int] = []
    match_counts: List[int] = []
    init_errors_rot: List[float] = []
    init_errors_trans: List[float] = []

    with torch.no_grad():
        for qi, entry in enumerate(retrieval_entries):
            query_sample = query_by_name[entry["query_image_name"]]
            used_query_samples.append(query_sample)
            query_path = _sample_image_path(images_root, entry["query_image_name"])
            refs = candidate_ref_entries_from_retrieval_entry(entry, topk=topk)
            candidate_results: List[Dict] = []
            candidate_corr_payloads: List[Dict | None] = []
            for ref in refs:
                ref_path = _sample_image_path(images_root, ref["retrieval_image_name"])
                fallback_pose = np.asarray(ref["fallback_pose_w2c"], dtype=np.float32)
                try:
                    ref_depth = _render_depth_for_loftr(
                        gaussians_depth,
                        fallback_pose,
                        intrinsics,
                        render_hw,
                        orig_hw,
                        device,
                    )
                    result = loftr.estimate_pose_direct(
                        query_path,
                        ref_path,
                        ref_depth,
                        fallback_pose,
                        intrinsics,
                        orig_hw,
                    )
                except Exception as exc:
                    class _Failed:
                        success = False
                        pose_w2c = None
                        num_inliers = 0
                        num_confident_matches = 0
                        failure_reason = f"exception:{type(exc).__name__}:{exc}"

                    result = _Failed()
                if bool(getattr(result, "success", False)) and getattr(result, "pose_w2c", None) is not None:
                    attach_pnp_quality_stats(result, result.pose_w2c, intr_loftr)
                cand = loftr_result_to_candidate(
                    result,
                    fallback_pose_w2c=fallback_pose,
                    retrieval_frame_id=ref["retrieval_frame_id"],
                    retrieval_image_name=ref["retrieval_image_name"],
                    retrieval_score=ref["retrieval_score"],
                )
                candidate_results.append(cand)
                candidate_corr_payloads.append(
                    _teacher_correspondence_payload(
                        result,
                        query_pose_w2c=query_sample["pose_w2c"],
                        intrinsics_loftr=intr_loftr,
                        loftr_hw=loftr_hw,
                        max_points=int(correspondence_max_points),
                        inliers_only=bool(correspondence_inliers_only),
                    )
                    if cand["pnp_success"]
                    else None
                )
                if cand["pnp_success"]:
                    inlier_counts.append(int(cand["num_inliers"]))
                    match_counts.append(int(cand["num_matches"]))
                    rot, trans = _pose_error(cand["pose_w2c"], query_sample["pose_w2c"])
                    init_errors_rot.append(rot)
                    init_errors_trans.append(trans)
                else:
                    reason = cand.get("failure_reason") or "unknown"
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
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
                candidate_corr_payloads.append(None)
            if corr_dir is not None and candidate_results:
                best_idx = max(
                    range(len(candidate_results)),
                    key=lambda idx: (
                        1 if candidate_results[idx].get("pnp_success") else 0,
                        int(candidate_results[idx].get("num_inliers", 0)),
                        float(candidate_results[idx].get("retrieval_score", 0.0)),
                    ),
                )
                payload = candidate_corr_payloads[best_idx]
                if payload is not None:
                    np.savez_compressed(corr_dir / f"{_feature_stem(entry['query_image_name'])}.npz", **payload)
            candidate_results_by_query.append(candidate_results)
            if (qi + 1) % 5 == 0 or qi == 0:
                successes = sum(1 for group in candidate_results_by_query for c in group if c.get("pnp_success"))
                print(f"[{qi + 1}/{len(retrieval_entries)}] render-LoFTR PnP success candidates: {successes}")

    entries, stats = build_pairwise_pnp_entries(
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
            "ply_path": str(ply_path),
            "orig_hw": list(orig_hw),
            "render_hw": list(render_hw),
            "loftr_hw": list(loftr_hw),
            "loftr_long_edge": int(loftr_long_edge),
            "loftr_conf": float(loftr_conf),
            "reproj_threshold": float(reproj_threshold),
            "pnp_iters": int(pnp_iters),
            "query_start": int(query_start),
            "max_queries": int(max_queries),
            "failure_counts": failure_counts,
            "mean_success_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
            "mean_success_matches": float(np.mean(match_counts)) if match_counts else 0.0,
            "success_pose_rot_median": float(np.median(init_errors_rot)) if init_errors_rot else None,
            "success_pose_trans_median": float(np.median(init_errors_trans)) if init_errors_trans else None,
            "num_exported_entries": int(len(entries)),
            "teacher_correspondence_dir": str(corr_dir) if corr_dir is not None else None,
            "teacher_correspondence_max_points": int(correspondence_max_points),
            "teacher_correspondence_inliers_only": bool(correspondence_inliers_only),
        }
    )
    save_retrieval_init_entries(entries, stats, save_path)
    del loftr, gaussians_depth
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retrieval-guided render-depth LoFTR+PnP init cache")
    parser.add_argument("--retrieval_init_path", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--source_name", default="netvlad_render_loftr_pnp")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loftr_long_edge", type=int, default=840)
    parser.add_argument("--loftr_conf", type=float, default=0.3)
    parser.add_argument("--reproj_threshold", type=float, default=8.0)
    parser.add_argument("--pnp_iters", type=int, default=10000)
    parser.add_argument("--render_long_edge", type=int, default=840)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--use_magsac", action="store_true")
    parser.add_argument("--save_correspondence_dir", default=None)
    parser.add_argument("--correspondence_max_points", type=int, default=512)
    parser.add_argument("--correspondence_inliers_only", action="store_true", default=True)
    parser.add_argument("--correspondence_all_depth_valid", action="store_false", dest="correspondence_inliers_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_render_loftr_pnp_init(
        retrieval_init_path=args.retrieval_init_path,
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        images_root=args.images_root,
        ply_path=args.ply_path,
        save_path=args.save_path,
        topk=args.topk,
        source_name=args.source_name,
        gpu=args.gpu,
        loftr_long_edge=args.loftr_long_edge,
        loftr_conf=args.loftr_conf,
        reproj_threshold=args.reproj_threshold,
        pnp_iters=args.pnp_iters,
        render_long_edge=args.render_long_edge,
        query_start=args.query_start,
        max_queries=args.max_queries,
        use_magsac=args.use_magsac,
        save_correspondence_dir=args.save_correspondence_dir,
        correspondence_max_points=args.correspondence_max_points,
        correspondence_inliers_only=args.correspondence_inliers_only,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
