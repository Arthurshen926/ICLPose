#!/usr/bin/env python3
"""Export LoFTR correspondences against GT-pose renders for NVS Stage 1.

The older correspondence caches were produced against retrieval/LoFTR initial
renders.  NVS Stage-1 trains against the GT render, so the map-side keypoints
must live in that same rendered view.  This tool renders the 3DGS map at each
query GT pose, matches real query RGB to the GT render with LoFTR, and writes a
TeacherCorrespondenceStore-compatible ``.npz`` per query.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import camera_params_to_intrinsics, read_colmap_cameras  # noqa: E402
from data.radio_loc_retrieval_dataset import list_colmap_split_samples  # noqa: E402
from pose_refine.evaluate_pipeline import _render_rgbd_for_loftr  # noqa: E402
from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution  # noqa: E402


def _sample_stem(image_name: str) -> str:
    return str(image_name).replace("\\", "/").replace("/", "_").rsplit(".", 1)[0]


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def payload_from_loftr_result(
    result,
    *,
    max_points: int = 512,
    inlier_only: bool = True,
    source: str = "gt_render_loftr",
) -> Dict[str, np.ndarray]:
    """Build a TeacherCorrespondenceStore payload from a LoFTRResult."""
    extra = getattr(result, "extra", {}) or {}
    query_xy = np.asarray(extra.get("query_keypoints", np.zeros((0, 2))), dtype=np.float32)
    map_xy = np.asarray(extra.get("ref_keypoints", np.zeros((0, 2))), dtype=np.float32)
    confidence = np.asarray(extra.get("confidence", np.zeros((0,))), dtype=np.float32)
    pts3d = np.asarray(extra.get("pts3d_world", np.zeros((len(query_xy), 3))), dtype=np.float32)
    if query_xy.ndim != 2 or query_xy.shape[-1] != 2:
        query_xy = np.zeros((0, 2), dtype=np.float32)
    if map_xy.ndim != 2 or map_xy.shape[-1] != 2:
        map_xy = np.zeros((0, 2), dtype=np.float32)
    count = min(len(query_xy), len(map_xy), len(confidence), len(pts3d))
    query_xy = query_xy[:count]
    map_xy = map_xy[:count]
    confidence = confidence[:count]
    pts3d = pts3d[:count]

    if bool(inlier_only) and count > 0:
        inliers = np.asarray(extra.get("pnp_inlier_mask", np.ones((count,), dtype=bool)), dtype=bool)[:count]
        if int(inliers.sum()) > 0:
            query_xy = query_xy[inliers]
            map_xy = map_xy[inliers]
            confidence = confidence[inliers]
            pts3d = pts3d[inliers]

    if len(confidence) > 0:
        order = np.argsort(-confidence)
        if int(max_points) > 0:
            order = order[: int(max_points)]
        query_xy = query_xy[order]
        map_xy = map_xy[order]
        confidence = confidence[order]
        pts3d = pts3d[order]

    loftr_hw = np.asarray(extra.get("loftr_hw", [0, 0]), dtype=np.int32).reshape(-1)
    if loftr_hw.size < 2:
        loftr_hw = np.asarray([0, 0], dtype=np.int32)
    else:
        loftr_hw = loftr_hw[:2]
    return {
        "query_xy": query_xy.astype(np.float32),
        "map_xy": map_xy.astype(np.float32),
        "pts3d_world": pts3d.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "query_hw": loftr_hw.astype(np.int32),
        "map_hw": loftr_hw.astype(np.int32),
        "coordinate_space": np.asarray("image"),
        "source": np.asarray(str(source)),
    }


def export_gt_render_loftr_correspondences(
    *,
    colmap_dir: str,
    query_split: str,
    images_root: str,
    ply_path: str,
    save_dir: str,
    gpu: int = 0,
    loftr_long_edge: int = 840,
    loftr_conf: float = 0.3,
    reproj_threshold: float = 8.0,
    pnp_iters: int = 10000,
    render_long_edge: int = 640,
    query_start: int = 0,
    max_queries: int = 0,
    max_points: int = 512,
    inlier_only: bool = True,
) -> Dict:
    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    samples = list_colmap_split_samples(colmap_dir, query_split)
    start = max(0, int(query_start))
    samples = samples[start:]
    if int(max_queries or 0) > 0:
        samples = samples[: int(max_queries)]

    cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    intrinsics = camera_params_to_intrinsics(first_cam)
    orig_hw = (int(first_cam.height), int(first_cam.width))
    scale = float(render_long_edge) / float(max(orig_hw))
    render_hw = (
        max(8, int(round(orig_hw[0] * scale / 8.0)) * 8),
        max(8, int(round(orig_hw[1] * scale / 8.0)) * 8),
    )
    loftr_hw = _compute_loftr_resolution(orig_hw, int(loftr_long_edge))

    from feature_gaussian import HybridGaussianModel

    gaussians_depth = HybridGaussianModel(sh_degree=3, latent_dim=0)
    gaussians_depth.load_ply(ply_path, freeze_geometry=True)
    gaussians_depth.active_sh_degree = 3
    gaussians_depth = gaussians_depth.to(device) if hasattr(gaussians_depth, "to") else gaussians_depth

    loftr = LoFTRInitializer(
        device=device,
        loftr_long_edge=int(loftr_long_edge),
        confidence_threshold=float(loftr_conf),
        reproj_threshold=float(reproj_threshold),
        pnp_iters=int(pnp_iters),
    )

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    successes = 0
    with torch.no_grad():
        for local_idx, sample in enumerate(samples):
            query_rgb = _load_rgb(os.path.join(images_root, sample["image_name"]))
            rendered_rgb, rendered_depth = _render_rgbd_for_loftr(
                gaussians_depth,
                np.asarray(sample["pose_w2c"], dtype=np.float32),
                intrinsics,
                render_hw,
                orig_hw,
                device,
            )
            result = loftr.estimate_pose(
                query_rgb,
                rendered_rgb,
                rendered_depth,
                np.asarray(sample["pose_w2c"], dtype=np.float32),
                intrinsics,
                orig_hw,
                confidence_threshold=float(loftr_conf),
                min_matches=4,
                reproj_threshold=float(reproj_threshold),
            )
            payload = payload_from_loftr_result(
                result,
                max_points=int(max_points),
                inlier_only=bool(inlier_only),
                source="gt_render_loftr",
            )
            out_path = out_dir / f"{_sample_stem(sample['image_name'])}.npz"
            np.savez(out_path, **payload)
            count = int(len(payload["confidence"]))
            counts.append(count)
            successes += int(count > 0)
            if local_idx == 0 or (local_idx + 1) % 4 == 0:
                print(f"[{local_idx + 1}/{len(samples)}] GT-render LoFTR points={count}", flush=True)

    stats = {
        "query_split": str(query_split),
        "num_queries": int(len(samples)),
        "queries_with_points": int(successes),
        "mean_points": float(np.mean(counts)) if counts else 0.0,
        "median_points": float(np.median(counts)) if counts else 0.0,
        "loftr_long_edge": int(loftr_long_edge),
        "loftr_hw": list(map(int, loftr_hw)),
        "loftr_conf": float(loftr_conf),
        "render_hw": list(map(int, render_hw)),
        "inlier_only": bool(inlier_only),
        "max_points": int(max_points),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    del loftr, gaussians_depth
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--query-split", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loftr-long-edge", type=int, default=840)
    parser.add_argument("--loftr-conf", type=float, default=0.3)
    parser.add_argument("--reproj-threshold", type=float, default=8.0)
    parser.add_argument("--pnp-iters", type=int, default=10000)
    parser.add_argument("--render-long-edge", type=int, default=640)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--all-depth-valid-matches", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_gt_render_loftr_correspondences(
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        images_root=args.images_root,
        ply_path=args.ply_path,
        save_dir=args.save_dir,
        gpu=args.gpu,
        loftr_long_edge=args.loftr_long_edge,
        loftr_conf=args.loftr_conf,
        reproj_threshold=args.reproj_threshold,
        pnp_iters=args.pnp_iters,
        render_long_edge=args.render_long_edge,
        query_start=args.query_start,
        max_queries=args.max_queries,
        max_points=args.max_points,
        inlier_only=not bool(args.all_depth_valid_matches),
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

