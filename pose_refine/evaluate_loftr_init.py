#!/usr/bin/env python3
"""Canonical PoseRefine LoFTR-init evaluation entrypoint.

Evaluate LoFTR-based sparse pose initialisation on Cambridge Landmarks.

For each test image the script finds the *K* nearest training images (by
camera-centre distance), renders depth at each reference pose via the
Gaussian splat model, runs LoFTR matching + PnP, and reports rotation /
translation error statistics.

Example::

    python -m scripts.eval_loftr_init \\
        --scene_root dataset/OldHospital \\
        --ply_path output/2dgs_models/OldHospital/point_cloud/iteration_30000/point_cloud.ply \\
        --num_neighbors 3 \\
        --loftr_resolution 840 \\
        --device cuda:0

If ``--ply_path`` is omitted the script falls back to **direct-image mode**
which matches against the actual training images and uses pre-rendered depth
maps from ``--depth_dir`` instead of live Gaussian rendering.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import struct
import sys
import time
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Add project root to path
_PROJ_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_loftr_init")


# ═══════════════════════════════════════════════════════════════════════════════
#  COLMAP binary readers (self-contained to avoid dataset import side-effects)
# ═══════════════════════════════════════════════════════════════════════════════

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageMeta = collections.namedtuple("ImageMeta", ["id", "qvec", "tvec", "camera_id", "name"])


def _read_cameras_bin(path: str) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("I", f.read(4))[0]
            mid = struct.unpack("i", f.read(4))[0]
            w = struct.unpack("Q", f.read(8))[0]
            h = struct.unpack("Q", f.read(8))[0]
            np_ = CAMERA_MODELS[mid].num_params
            params = np.array(struct.unpack(f"{np_}d", f.read(8 * np_)))
            cameras[cid] = Camera(cid, CAMERA_MODELS[mid].model_name, w, h, params)
    return cameras


def _read_images_bin(path: str) -> Dict[int, ImageMeta]:
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("I", f.read(4))[0]
            qvec = np.array(struct.unpack("4d", f.read(32)))
            tvec = np.array(struct.unpack("3d", f.read(24)))
            cid = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            num_pts = struct.unpack("Q", f.read(8))[0]
            f.read(num_pts * 24)
            images[iid] = ImageMeta(iid, qvec, tvec, cid, name)
    return images


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def _colmap_to_w2c(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = _qvec_to_rotmat(qvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def _camera_center(w2c: np.ndarray) -> np.ndarray:
    """Extract camera centre in world frame: C = -R^T t."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return -R.T @ t


def _cam_params_to_intr(cam: Camera) -> Dict[str, float]:
    p = cam.params
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_split_file(path: str) -> set:
    """Parse Cambridge Landmarks split file → set of image name stems."""
    names = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            parts = line.split()
            if parts:
                base = os.path.splitext(parts[0])[0]
                names.add(base + ".png")
                names.add(base + ".jpg")
    return names


def _build_sample_list(
    colmap_dir: str,
    split_file: str,
    scene_root: str,
) -> Tuple[List[Dict], Dict[str, float], Tuple[int, int]]:
    """Build a list of {name, img_path, pose_w2c, camera_center} from COLMAP + split."""
    cameras = _read_cameras_bin(os.path.join(colmap_dir, "cameras.bin"))
    images = _read_images_bin(os.path.join(colmap_dir, "images.bin"))
    split_names = _parse_split_file(split_file)

    samples: List[Dict] = []
    intrinsics = None
    orig_hw = None

    for img_id in sorted(images.keys()):
        meta = images[img_id]
        if meta.name not in split_names:
            continue
        w2c = _colmap_to_w2c(meta.qvec, meta.tvec).astype(np.float32)
        center = _camera_center(w2c)

        img_path = os.path.join(scene_root, meta.name)

        cam = cameras[meta.camera_id]
        if intrinsics is None:
            intrinsics = _cam_params_to_intr(cam)
            orig_hw = (int(cam.height), int(cam.width))

        samples.append({
            "img_id": img_id,
            "name": meta.name,
            "img_path": img_path,
            "pose_w2c": w2c,
            "camera_center": center,
        })

    return samples, intrinsics, orig_hw


# ═══════════════════════════════════════════════════════════════════════════════
#  Nearest-neighbor oracle
# ═══════════════════════════════════════════════════════════════════════════════

def _find_nearest_train(
    query_sample: Dict,
    train_samples: List[Dict],
    train_centers: np.ndarray,
    K: int,
) -> List[int]:
    """Return indices of the K nearest training samples by camera-centre distance."""
    qc = query_sample["camera_center"]
    dists = np.linalg.norm(train_centers - qc[None], axis=1)
    return list(np.argsort(dists)[:K])


# ═══════════════════════════════════════════════════════════════════════════════
#  Error metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices (degrees)."""
    R_rel = R_pred.T @ R_gt
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return float(np.degrees(np.arccos(cos_angle)))


def _translation_error_m(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Camera-center distance in metres."""
    return float(np.linalg.norm(t_pred - t_gt))


def _pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> Tuple[float, float]:
    """Return (rot_err_deg, trans_err_m) between two w2c matrices.

    Translation error is measured as the distance between camera centres
    in world frame (not the raw w2c translation vectors).
    """
    rot_err = _rotation_error_deg(pred_w2c[:3, :3], gt_w2c[:3, :3])
    c_pred = _camera_center(pred_w2c)
    c_gt = _camera_center(gt_w2c)
    trans_err = float(np.linalg.norm(c_pred - c_gt))
    return rot_err, trans_err


# ═══════════════════════════════════════════════════════════════════════════════
#  Gaussian depth rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gaussian_renderer(ply_path: str, device: torch.device):
    """Load Gaussian model for RGB+depth rendering (no DCFF features)."""
    from feature_gaussian import HybridGaussianModel

    gaussians = HybridGaussianModel(sh_degree=3, latent_dim=0)
    gaussians.load_ply(ply_path, freeze_geometry=True)
    gaussians.active_sh_degree = 3
    logger.info("Loaded %d Gaussians from %s", gaussians.num_points, ply_path)
    return gaussians


@torch.no_grad()
def _render_depth(
    gaussians,
    pose_w2c: np.ndarray,
    intrinsics: Dict[str, float],
    render_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """Render depth at *render_hw* resolution. Returns (H, W) numpy float32."""
    from gsplat.rendering import rasterization_2dgs

    intr_scaled = {
        "fx": intrinsics["fx"] * render_hw[1] / orig_hw[1],
        "fy": intrinsics["fy"] * render_hw[0] / orig_hw[0],
        "cx": intrinsics["cx"] * render_hw[1] / orig_hw[1],
        "cy": intrinsics["cy"] * render_hw[0] / orig_hw[0],
    }
    K_mat = torch.tensor([
        [intr_scaled["fx"], 0, intr_scaled["cx"]],
        [0, intr_scaled["fy"], intr_scaled["cy"]],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)

    viewmat = torch.from_numpy(pose_w2c.astype(np.float32)).to(device).unsqueeze(0)

    means3d = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling_for_render  # [N, 3] padded for 2DGS
    opacities = gaussians.get_opacity.squeeze(-1)
    colors = gaussians.get_features  # SH coefficients

    rH, rW = render_hw
    _colors, _alphas, _normals, _surf_normals, _distort, _median, meta = rasterization_2dgs(
        means=means3d,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K_mat,
        width=rW,
        height=rH,
        packed=False,
        near_plane=0.01,
        far_plane=1e5,
        render_mode="RGB+ED",
        sh_degree=gaussians.active_sh_degree,
    )
    # _colors shape: (1, H, W, (3+1)) — last channel is expected depth
    depth = _colors[0, :, :, 3].cpu().numpy()
    return depth


@torch.no_grad()
def _render_rgb_depth(
    gaussians,
    pose_w2c: np.ndarray,
    intrinsics: Dict[str, float],
    render_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render RGB + depth. Returns ((H,W,3) uint8, (H,W) float32)."""
    from gsplat.rendering import rasterization_2dgs

    intr_scaled = {
        "fx": intrinsics["fx"] * render_hw[1] / orig_hw[1],
        "fy": intrinsics["fy"] * render_hw[0] / orig_hw[0],
        "cx": intrinsics["cx"] * render_hw[1] / orig_hw[1],
        "cy": intrinsics["cy"] * render_hw[0] / orig_hw[0],
    }
    K_mat = torch.tensor([
        [intr_scaled["fx"], 0, intr_scaled["cx"]],
        [0, intr_scaled["fy"], intr_scaled["cy"]],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)

    viewmat = torch.from_numpy(pose_w2c.astype(np.float32)).to(device).unsqueeze(0)

    means3d = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling_for_render  # [N, 3] padded for 2DGS
    opacities = gaussians.get_opacity.squeeze(-1)
    colors = gaussians.get_features

    rH, rW = render_hw
    render_out, _alphas, _normals, _surf_normals, _distort, _median, meta = rasterization_2dgs(
        means=means3d,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K_mat,
        width=rW,
        height=rH,
        packed=False,
        near_plane=0.01,
        far_plane=1e5,
        render_mode="RGB+ED",
        sh_degree=gaussians.active_sh_degree,
    )
    rgb = render_out[0, :, :, :3].clamp(0, 1).cpu().numpy()
    rgb_u8 = (rgb * 255).astype(np.uint8)
    depth = render_out[0, :, :, 3].cpu().numpy()
    return rgb_u8, depth


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    scene_root = args.scene_root
    colmap_dir = os.path.join(scene_root, "sparse", "0")
    train_split = os.path.join(scene_root, "dataset_train.txt")
    test_split = os.path.join(scene_root, "dataset_test.txt")

    # Build sample lists
    logger.info("Loading train/test samples from %s …", scene_root)
    train_samples, intrinsics, orig_hw = _build_sample_list(colmap_dir, train_split, scene_root)
    test_samples, _, _ = _build_sample_list(colmap_dir, test_split, scene_root)
    if args.max_test:
        test_samples = test_samples[: args.max_test]
    logger.info("Train: %d   Test: %d   orig_hw: %s", len(train_samples), len(test_samples), orig_hw)
    logger.info("Intrinsics: %s", intrinsics)

    train_centers = np.array([s["camera_center"] for s in train_samples])

    # Gaussian model (optional — for depth rendering)
    gaussians = None
    if args.ply_path and os.path.isfile(args.ply_path):
        gaussians = _build_gaussian_renderer(args.ply_path, device)

    # Decide rendering resolution for depth
    from pose_refine.sparse_init import _compute_loftr_resolution
    loftr_hw = _compute_loftr_resolution(orig_hw, args.loftr_resolution)
    if args.depth_render_hw:
        depth_hw = tuple(args.depth_render_hw)
    else:
        depth_hw = loftr_hw
    logger.info("LoFTR hw: %s   Depth render hw: %s", loftr_hw, depth_hw)

    # LoFTR initializer
    from pose_refine.sparse_init import LoFTRInitializer
    initializer = LoFTRInitializer(
        device=device,
        pretrained=args.loftr_pretrained,
        loftr_long_edge=args.loftr_resolution,
        confidence_threshold=args.confidence_threshold,
        min_matches=args.min_matches,
        reproj_threshold=args.reproj_threshold,
    )

    mode = args.mode  # "direct" or "render"
    use_direct = mode == "direct"
    if use_direct and gaussians is None and not args.depth_dir:
        logger.error("Direct mode requires either --ply_path (for live depth) or --depth_dir")
        sys.exit(1)

    rot_errors: List[float] = []
    trans_errors: List[float] = []
    init_rot_errors: List[float] = []
    init_trans_errors: List[float] = []
    n_failed = 0
    match_stats: List[Dict] = []

    t0 = time.time()
    for qi, query in enumerate(test_samples):
        # Find K nearest training images
        nn_indices = _find_nearest_train(query, train_samples, train_centers, args.num_neighbors)

        best_result = None
        for ni in nn_indices:
            ref = train_samples[ni]

            # Report baseline (retrieval) error
            init_rot, init_trans = _pose_errors(ref["pose_w2c"], query["pose_w2c"])

            # Get depth for reference pose
            if args.depth_dir:
                depth_path = os.path.join(args.depth_dir, f"depth_{ref['img_id']}.pt")
                if os.path.isfile(depth_path):
                    ref_depth = torch.load(depth_path, map_location="cpu").numpy()
                    if ref_depth.ndim == 3:
                        ref_depth = ref_depth[0]
                else:
                    ref_depth = None
            elif gaussians is not None:
                ref_depth = _render_depth(
                    gaussians, ref["pose_w2c"], intrinsics, depth_hw, orig_hw, device,
                )
            else:
                ref_depth = None

            if ref_depth is None:
                continue

            if use_direct:
                result = initializer.estimate_pose_direct(
                    query["img_path"], ref["img_path"],
                    ref_depth, ref["pose_w2c"], intrinsics, orig_hw,
                )
            else:
                # Render RGB at reference pose
                ref_rgb, ref_depth_rendered = _render_rgb_depth(
                    gaussians, ref["pose_w2c"], intrinsics, depth_hw, orig_hw, device,
                )
                result = initializer.estimate_pose(
                    cv2.cvtColor(cv2.imread(str(query["img_path"])), cv2.COLOR_BGR2RGB),
                    ref_rgb, ref_depth_rendered,
                    ref["pose_w2c"], intrinsics, orig_hw,
                )

            if best_result is None or result.num_inliers > best_result.num_inliers:
                best_result = result
                best_init_rot = init_rot
                best_init_trans = init_trans

        if best_result is not None and best_result.success:
            rot_e, trans_e = _pose_errors(best_result.pose_w2c, query["pose_w2c"])
            rot_errors.append(rot_e)
            trans_errors.append(trans_e)
            init_rot_errors.append(best_init_rot)
            init_trans_errors.append(best_init_trans)
            match_stats.append({
                "matches": best_result.num_confident_matches,
                "inliers": best_result.num_inliers,
                "conf": best_result.mean_confidence,
            })
        else:
            n_failed += 1
            reason = best_result.failure_reason if best_result else "no_depth"
            if qi < 10 or qi % 50 == 0:
                logger.warning("  [%d/%d] FAILED: %s", qi, len(test_samples), reason)

        if (qi + 1) % 20 == 0 or qi == len(test_samples) - 1:
            elapsed = time.time() - t0
            n_ok = len(rot_errors)
            if n_ok > 0:
                logger.info(
                    "  [%d/%d]  %.1fs  rot=%.2f°/%.2f°  trans=%.3fm/%.3fm  "
                    "failed=%d  matches=%.0f  inliers=%.0f",
                    qi + 1, len(test_samples), elapsed,
                    np.median(rot_errors), np.mean(rot_errors),
                    np.median(trans_errors), np.mean(trans_errors),
                    n_failed,
                    np.mean([s["matches"] for s in match_stats]),
                    np.mean([s["inliers"] for s in match_stats]),
                )

    # ── Final report ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_ok = len(rot_errors)
    n_total = len(test_samples)

    print("\n" + "=" * 70)
    print(f"LoFTR Sparse Init — {scene_root}")
    print(f"Mode: {mode}   Neighbors: {args.num_neighbors}   "
          f"LoFTR res: {loftr_hw}   Depth res: {depth_hw}")
    print(f"Confidence thr: {args.confidence_threshold}   "
          f"Reproj thr: {args.reproj_threshold}")
    print("=" * 70)
    print(f"Total: {n_total}   Success: {n_ok}   Failed: {n_failed}  "
          f"({n_ok/n_total*100:.1f}% success)")
    print(f"Time: {elapsed:.1f}s  ({elapsed/n_total:.2f}s/image)")

    if n_ok > 0:
        rot_arr = np.array(rot_errors)
        trans_arr = np.array(trans_errors)
        init_rot_arr = np.array(init_rot_errors)
        init_trans_arr = np.array(init_trans_errors)

        print(f"\n{'Metric':<30} {'Median':>10} {'Mean':>10} {'Std':>10}")
        print("-" * 60)
        print(f"{'Retrieval rot (°)':<30} {np.median(init_rot_arr):>10.2f} "
              f"{np.mean(init_rot_arr):>10.2f} {np.std(init_rot_arr):>10.2f}")
        print(f"{'Retrieval trans (m)':<30} {np.median(init_trans_arr):>10.3f} "
              f"{np.mean(init_trans_arr):>10.3f} {np.std(init_trans_arr):>10.3f}")
        print(f"{'LoFTR init rot (°)':<30} {np.median(rot_arr):>10.2f} "
              f"{np.mean(rot_arr):>10.2f} {np.std(rot_arr):>10.2f}")
        print(f"{'LoFTR init trans (m)':<30} {np.median(trans_arr):>10.3f} "
              f"{np.mean(trans_arr):>10.3f} {np.std(trans_arr):>10.3f}")

        # Percentile table
        print(f"\n{'Percentile':<20} {'Rot (°)':>10} {'Trans (m)':>10}")
        print("-" * 40)
        for pct in [25, 50, 75, 90, 95, 99]:
            print(f"{'p' + str(pct):<20} {np.percentile(rot_arr, pct):>10.2f} "
                  f"{np.percentile(trans_arr, pct):>10.3f}")

        # Threshold table
        print(f"\n{'Threshold':<30} {'LoFTR init':>12} {'Retrieval':>12}")
        print("-" * 55)
        for rdeg, tm in [(1.0, 0.05), (2.0, 0.10), (5.0, 0.25), (10.0, 0.50)]:
            pct_loftr = float(np.mean((rot_arr < rdeg) & (trans_arr < tm))) * 100
            pct_retr = float(np.mean((init_rot_arr < rdeg) & (init_trans_arr < tm))) * 100
            print(f"{'< ' + str(rdeg) + '° / ' + str(tm) + 'm':<30} "
                  f"{pct_loftr:>11.1f}% {pct_retr:>11.1f}%")

        # Match statistics
        matches = np.array([s["matches"] for s in match_stats])
        inliers = np.array([s["inliers"] for s in match_stats])
        confs = np.array([s["conf"] for s in match_stats])
        print(f"\nMatching statistics (successful only):")
        print(f"  Confident matches: {np.median(matches):.0f} med / {np.mean(matches):.0f} mean")
        print(f"  PnP inliers:      {np.median(inliers):.0f} med / {np.mean(inliers):.0f} mean")
        print(f"  Mean confidence:   {np.mean(confs):.3f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate LoFTR-based sparse pose initialisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene_root", required=True,
                    help="Path to scene directory (e.g. dataset/OldHospital)")
    p.add_argument("--ply_path", default=None,
                    help="Path to Gaussian .ply for live depth rendering")
    p.add_argument("--depth_dir", default=None,
                    help="Directory with pre-rendered depth_{img_id}.pt files")
    p.add_argument("--mode", default="direct", choices=["direct", "render"],
                    help="'direct': match real images; 'render': match rendered RGB")
    p.add_argument("--num_neighbors", type=int, default=3,
                    help="Number of nearest training images to try")
    p.add_argument("--loftr_resolution", type=int, default=840,
                    help="Target long-edge for LoFTR input (aspect-preserving)")
    p.add_argument("--loftr_pretrained", default="outdoor",
                    help="LoFTR pretrained variant")
    p.add_argument("--confidence_threshold", type=float, default=0.3,
                    help="Minimum LoFTR match confidence")
    p.add_argument("--min_matches", type=int, default=6,
                    help="Minimum confident matches required")
    p.add_argument("--reproj_threshold", type=float, default=8.0,
                    help="PnP RANSAC reprojection error threshold (px)")
    p.add_argument("--depth_render_hw", type=int, nargs=2, default=None,
                    metavar=("H", "W"),
                    help="Depth render resolution (default: same as LoFTR)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_test", type=int, default=None,
                    help="Limit number of test images (for debugging)")

    args = p.parse_args()

    if args.max_test:
        # Handled via slicing after loading
        pass

    run_evaluation(args)


if __name__ == "__main__":
    main()
