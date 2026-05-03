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
from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution  # noqa: E402


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
        "failure_reason": "" if success else str(getattr(result, "failure_reason", "")),
    }


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
    max_queries: int = 0,
    use_magsac: bool = False,
) -> Dict:
    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    query_samples_all = list_colmap_split_samples(colmap_dir, query_split)
    query_by_name = {sample["image_name"]: sample for sample in query_samples_all}
    retrieval_entries, retrieval_stats = load_retrieval_init_entries(retrieval_init_path)
    retrieval_entries = [entry for entry in retrieval_entries if entry["query_image_name"] in query_by_name]
    if max_queries and int(max_queries) > 0:
        retrieval_entries = retrieval_entries[: int(max_queries)]

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
                cand = loftr_result_to_candidate(
                    result,
                    fallback_pose_w2c=fallback_pose,
                    retrieval_frame_id=ref["retrieval_frame_id"],
                    retrieval_image_name=ref["retrieval_image_name"],
                    retrieval_score=ref["retrieval_score"],
                )
                candidate_results.append(cand)
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
            "failure_counts": failure_counts,
            "mean_success_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
            "mean_success_matches": float(np.mean(match_counts)) if match_counts else 0.0,
            "success_pose_rot_median": float(np.median(init_errors_rot)) if init_errors_rot else None,
            "success_pose_trans_median": float(np.median(init_errors_trans)) if init_errors_trans else None,
            "num_exported_entries": int(len(entries)),
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
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--use_magsac", action="store_true")
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
        max_queries=args.max_queries,
        use_magsac=args.use_magsac,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
