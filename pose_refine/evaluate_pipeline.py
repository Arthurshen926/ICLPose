#!/usr/bin/env python3
"""Canonical PoseRefine pipeline evaluation entrypoint.

End-to-end pipeline evaluation: retrieval → LoFTR init → NN refinement →
multi-start selection.

For each test image:
  0. Oracle retrieval – find K nearest training images by camera-centre distance
  1. LoFTR sparse init – estimate pose from each of K neighbours
  2. ConcatPoseNet refinement – iteratively refine *each* candidate
  3. Multi-start selection – pick best refined pose by feature residual

Reports metrics at every stage and saves a JSON results file.

Example::

    python -m pose_refine.evaluate_pipeline \\
    --config pose_refine/configs/concat_loc_oh_v22d_stage2_lownoise.yaml \
    --checkpoint pose_refine/output/concat_loc_oh_v22d_stage2_lownoise/checkpoints/best.pth \
        --gpu 0 --outer_iters 5 --gru_iters 6 --num_neighbors 3
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

_PROJ_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from feature_field.utils.loc_reporting import save_experiment_bundle


def _setup_gpu(gpu: int) -> None:
    """Set CUDA_VISIBLE_DEVICES *before* torch is imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


# ═══════════════════════════════════════════════════════════════════════════════
#  COLMAP binary readers (minimal, self-contained)
# ═══════════════════════════════════════════════════════════════════════════════

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
}
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def _read_cameras_binary(path: str) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<IiQQ", f.read(24))
            nparams = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
            cameras[cam_id] = Camera(cam_id, model_id, w, h, params)
    return cameras


def _read_images_binary(path: str) -> Dict[int, BaseImage]:
    images: Dict[int, BaseImage] = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts * 24)  # skip 2D point data (16 xy + 8 id each)
            images[img_id] = BaseImage(img_id, qvec, tvec, cam_id, name, None, None)
    return images


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np


def _qvec_to_rotmat(qvec) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def _camera_center(w2c) -> np.ndarray:
    """C = -R^T t  (world-frame camera position)."""
    if hasattr(w2c, "numpy"):
        w2c = w2c.cpu().float().numpy()
    return -w2c[:3, :3].T @ w2c[:3, 3]


def _cam_params_to_intr(cam: Camera) -> Dict[str, float]:
    p = cam.params
    if cam.model == 1:  # PINHOLE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:  # SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def _parse_split_file(path: str) -> set:
    """Parse split file → set of image names (supports Cambridge and 7Scenes formats)."""
    names: set = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            parts = line.split()
            if not parts:
                continue
            img_name = parts[0]
            names.add(img_name)
            # Also add with swapped extension for robustness
            base = os.path.splitext(img_name)[0]
            names.add(base + ".png")
            names.add(base + ".jpg")
    return names


def _build_sample_list(
    colmap_dir: str,
    split_file: str,
    source_dir: str,
) -> Tuple[List[Dict], Dict[str, float], Tuple[int, int]]:
    cameras = _read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))
    images = _read_images_binary(os.path.join(colmap_dir, "images.bin"))
    split_names = _parse_split_file(split_file)

    first_cam = next(iter(cameras.values()))
    intrinsics = _cam_params_to_intr(first_cam)
    orig_hw = (int(first_cam.height), int(first_cam.width))

    samples: List[Dict] = []
    for img_id in sorted(images.keys()):
        meta = images[img_id]
        if meta.name not in split_names:
            continue
        R = _qvec_to_rotmat(meta.qvec)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = meta.tvec
        samples.append({
            "img_id": img_id,
            "img_name": meta.name,
            "img_path": os.path.join(source_dir, meta.name),
            "pose_w2c": w2c,
            "camera_center": _camera_center(w2c),
        })
    return samples, intrinsics, orig_hw


def _find_k_nearest(
    query_center: np.ndarray,
    train_centers: np.ndarray,
    K: int,
) -> List[int]:
    dists = np.linalg.norm(train_centers - query_center[None], axis=1)
    return list(np.argsort(dists)[:K])


def _resolve_eval_mode_metadata(eval_mode: Optional[str], init_method: str) -> Dict[str, Any]:
    """Resolve oracle/deploy labels and reject known mode mixing."""
    init_method = str(init_method or "loftr")
    if eval_mode is None:
        eval_mode = "oracle" if init_method == "loftr" else "deploy"
    eval_mode = str(eval_mode).lower()
    if eval_mode not in {"oracle", "deploy", "ablation"}:
        raise ValueError(f"Unsupported eval_mode={eval_mode!r}")
    uses_gt_query_center = init_method == "loftr"
    if eval_mode == "deploy" and uses_gt_query_center:
        raise ValueError(
            "eval_mode=deploy cannot be used with init_method=loftr because that path uses oracle "
            "GT query camera centers for nearest-neighbor retrieval. Use init_method=regression or "
            "regression_loftr, or mark the run as eval_mode=oracle/ablation."
        )
    return {
        "eval_mode": eval_mode,
        "init_method": init_method,
        "uses_gt_query_center": bool(uses_gt_query_center),
        "uses_gt_query_pose": bool(uses_gt_query_center),
        "deployable": bool(eval_mode == "deploy" and not uses_gt_query_center),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Pose error computation
# ═══════════════════════════════════════════════════════════════════════════════

def _pose_errors(pred_w2c, gt_w2c) -> Tuple[float, float]:
    """Return (rot_deg, trans_m). pred/gt can be numpy or torch (4,4)."""
    import torch
    if isinstance(pred_w2c, torch.Tensor):
        pred_w2c = pred_w2c.cpu().float().numpy()
    if isinstance(gt_w2c, torch.Tensor):
        gt_w2c = gt_w2c.cpu().float().numpy()
    R_rel = pred_w2c[:3, :3].T @ gt_w2c[:3, :3]
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    c_pred = _camera_center(pred_w2c)
    c_gt = _camera_center(gt_w2c)
    trans_m = float(np.linalg.norm(c_pred - c_gt))
    return rot_deg, trans_m


# ═══════════════════════════════════════════════════════════════════════════════
#  Gaussian depth rendering (for LoFTR — uses rasterization_2dgs directly)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_depth_for_loftr(gaussians, pose_w2c, intrinsics, render_hw, orig_hw, device):
    """Render depth via gsplat 2DGS. Returns (H,W) numpy float32."""
    _, depth = _render_rgbd_for_loftr(gaussians, pose_w2c, intrinsics, render_hw, orig_hw, device)
    return depth


def _render_rgbd_for_loftr(gaussians, pose_w2c, intrinsics, render_hw, orig_hw, device):
    """Render RGB+depth via gsplat 2DGS.

    Returns:
        rgb: (H, W, 3) numpy uint8  — clamped to [0, 255]
        depth: (H, W) numpy float32
    """
    import torch
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
    out, _a, _n, _sn, _d, _m, _meta = rasterization_2dgs(
        means=means3d, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmat, Ks=K_mat,
        width=rW, height=rH, packed=False,
        near_plane=0.01, far_plane=1e5,
        render_mode="RGB+ED", sh_degree=gaussians.active_sh_degree,
    )
    # out shape: (1, H, W, 3+1) — last channel is expected depth
    rgb_float = out[0, :, :, :3].detach().clamp(0.0, 1.0).cpu().numpy()
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)
    depth = out[0, :, :, 3].detach().cpu().numpy()
    return rgb_uint8, depth


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    import torch
    from torch.cuda.amp import autocast
    from scipy.spatial.transform import Rotation as R_scipy

    eval_mode_meta = _resolve_eval_mode_metadata(
        getattr(args, "eval_mode", None),
        getattr(args, "init_method", "loftr"),
    )
    args.eval_mode = eval_mode_meta["eval_mode"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    logger = logging.getLogger("pipeline_eval")
    if eval_mode_meta["uses_gt_query_center"]:
        logger.warning(
            "EVAL_MODE=%s uses GT query camera centers for retrieval; report this run as ORACLE/ABLATION, not deployable.",
            eval_mode_meta["eval_mode"].upper(),
        )

    def _perturb_pose(pose_w2c: np.ndarray, noise_deg: float, noise_m: float,
                      rng: np.random.RandomState) -> np.ndarray:
        """Add small random perturbation to a w2c pose."""
        pose = pose_w2c.copy().astype(np.float64)
        # Rotation perturbation
        angle_rad = np.deg2rad(noise_deg)
        axis = rng.randn(3)
        axis /= max(np.linalg.norm(axis), 1e-8)
        angle = rng.randn() * angle_rad
        dR = R_scipy.from_rotvec(axis * angle).as_matrix()
        pose[:3, :3] = dR @ pose[:3, :3]
        # Translation perturbation
        pose[:3, 3] += rng.randn(3) * noise_m
        return pose.astype(np.float32)

    # ── Load refinement config and models ─────────────────────────────────
    from feature_field.utils.project_config import load_mainline_config
    from feature_field import build_dcff, intrinsics_to_K, render_batch
    from feature_field.runtime import apply_localization_map_state
    from pose_refine import load_concat_pose_checkpoint, load_concat_pose_model as load_model
    from pose_refine.utils.lie_algebra import se3_exp

    config = load_mainline_config(args.config)
    logger.info("Building DCFF rendering pipeline...")
    gaussians_dcff, dcff_renderer, feat_sharp = build_dcff(config, device)

    logger.info("Loading refinement model...")
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    pose_ckpt = load_concat_pose_checkpoint(args.checkpoint, device)
    restored_map = apply_localization_map_state(
        dcff_renderer,
        feat_sharp,
        pose_ckpt,
        printer=logger.info,
    )
    if restored_map:
        logger.info("Restored localization map state: %s", ", ".join(restored_map))

    dcff_cfg = config["dcff"]
    render_h = dcff_cfg.get("render_height", 68)
    render_w = dcff_cfg.get("render_width", 120)
    use_coarse = config.get("model", {}).get("use_coarse", True)
    ds_cfg = config["dataset"]

    # Set intrinsics on model
    from data.radio_loc_dataset import read_colmap_cameras, camera_params_to_intrinsics
    colmap_dir = ds_cfg["colmap_dir"]
    colmap_cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
    first_cam = next(iter(colmap_cameras.values()))
    model.BASE_INTRINSICS = camera_params_to_intrinsics(first_cam)
    model.IMG_HW = (int(first_cam.height), int(first_cam.width))
    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    def _render_feature_bundle(pose_w2c_t: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        result = dcff_renderer(
            gaussians_dcff,
            viewmat=pose_w2c_t.float(),
            K=K,
            width=render_w,
            height=render_h,
            render_coarse=bool(getattr(dcff_renderer, "_fsm_use_coarse", False)),
            feature_height=render_h,
            feature_width=render_w,
        )
        from feature_field.runtime import _apply_dcff_postprocess
        result = _apply_dcff_postprocess(
            result,
            render_h,
            render_w,
            feat_sharp=feat_sharp,
            feat_select=getattr(dcff_renderer, "_feat_select", None),
            use_coarse_for_fsm=bool(getattr(dcff_renderer, "_fsm_use_coarse", False)),
            temperature=0.5,
            hard=False,
        )
        return {
            "fine_features": result["fine_features"].float(),
            "coarse_features": result.get("coarse_features").float() if result.get("coarse_features") is not None else None,
            "depth": result["depth"],
            "fsm_spatial_conf": result.get("fsm_spatial_conf").float() if result.get("fsm_spatial_conf") is not None else None,
        }

    def _feature_rerank_score(
        query_fine_t: torch.Tensor,
        query_coarse_t: Optional[torch.Tensor],
        render_bundle: Dict[str, Optional[torch.Tensor]],
        use_fsm_conf: bool,
    ) -> Tuple[float, float, float]:
        render_fine = render_bundle["fine_features"]
        render_coarse = render_bundle.get("coarse_features")
        fsm_conf = render_bundle.get("fsm_spatial_conf") if use_fsm_conf else None

        fine_sq = (query_fine_t - render_fine).pow(2)
        if fsm_conf is not None:
            fine_w = fsm_conf.expand_as(fine_sq)
            fine_score = float((fine_sq * fine_w).sum().item() / max(fine_w.sum().item(), 1e-6))
        else:
            fine_score = float(fine_sq.mean().item())

        coarse_score = 0.0
        if query_coarse_t is not None and render_coarse is not None:
            if render_coarse.shape[-2:] != query_coarse_t.shape[-2:]:
                render_coarse = torch.nn.functional.interpolate(
                    render_coarse,
                    size=query_coarse_t.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            coarse_sq = (query_coarse_t - render_coarse).pow(2)
            if fsm_conf is not None:
                coarse_conf = torch.nn.functional.interpolate(
                    fsm_conf,
                    size=query_coarse_t.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_w = coarse_conf.expand_as(coarse_sq)
                coarse_score = float((coarse_sq * coarse_w).sum().item() / max(coarse_w.sum().item(), 1e-6))
            else:
                coarse_score = float(coarse_sq.mean().item())

        return fine_score + coarse_score, fine_score, coarse_score

    # ── Load dataset for query features ───────────────────────────────────
    from data.radio_loc_dataset import RadioLocDataset
    fine_hw = tuple(ds_cfg.get("fine_hw", [render_h, render_w]))
    coarse_hw = tuple(ds_cfg.get("coarse_hw", fine_hw))
    val_ds = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        split_file=ds_cfg["test_split"],
        fine_hw=fine_hw,
        coarse_hw=coarse_hw,
        noise_rot_deg=1.0,  # not used — we override pose_init
        noise_trans_m=0.03,
        cache_in_memory=True,
    )

    # ── Build sample lists for LoFTR ──────────────────────────────────────
    scene_root = ds_cfg.get("source_dir", "dataset/OldHospital")
    colmap_sparse = os.path.join(scene_root, "sparse", "0")
    train_split = ds_cfg.get("train_split",
                             os.path.join(scene_root, "dataset_train.txt"))
    test_split = ds_cfg.get("test_split",
                            os.path.join(scene_root, "dataset_test.txt"))

    train_samples, intrinsics, orig_hw = _build_sample_list(
        colmap_sparse, train_split, scene_root)
    test_samples, _, _ = _build_sample_list(
        colmap_sparse, test_split, scene_root)
    if args.max_test:
        test_samples = test_samples[:args.max_test]
    train_centers = np.array([s["camera_center"] for s in train_samples])
    logger.info("Train: %d   Test: %d   orig_hw: %s",
                len(train_samples), len(test_samples), orig_hw)

    # Build name→dataset-index lookup for query features
    name_to_ds_idx: Dict[str, int] = {}
    from data.radio_loc_dataset import read_colmap_images
    colmap_all_images = read_colmap_images(os.path.join(colmap_dir, "images.bin"))
    for di in range(len(val_ds)):
        sample = val_ds[di]
        img_id = sample["image_id"]
        if isinstance(img_id, torch.Tensor):
            img_id = img_id.item()
        if img_id in colmap_all_images:
            name_to_ds_idx[colmap_all_images[img_id].name] = di
    logger.info("Dataset index: %d entries mapped", len(name_to_ds_idx))

    # ── Extract ply_path (needed for depth rendering in LoFTR methods) ───
    ply_path = dcff_cfg.get("ply_path",
                            config.get("renderer", {}).get("ply_path"))
    if ply_path is None:
        logger.error("No ply_path found in config")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: Initialisation (LoFTR or Regression)
    # ══════════════════════════════════════════════════════════════════════
    init_method = getattr(args, "init_method", "loftr")
    regression_npz_path = getattr(args, "regression_npz", None)

    # For 'inlier' selection: only refine the best init pose (compatible with all methods)
    loftr_candidates: List[List[Dict[str, Any]]] = []
    loftr_best: List[Dict[str, Any]] = []
    n_loftr_failed = 0

    if init_method == "regression":
        # ── Direct regression: load precomputed w2c poses, skip LoFTR ──
        assert regression_npz_path is not None, \
            "--regression_npz is required when --init_method=regression"
        reg_data = np.load(regression_npz_path, allow_pickle=True)
        reg_names = list(reg_data["img_names"])
        reg_poses = reg_data["poses_w2c"]      # (N, 4, 4) float32
        reg_centers = reg_data["camera_centers"]  # (N, 3)
        reg_name_to_idx = {n: i for i, n in enumerate(reg_names)}

        print(f"\n{'='*70}")
        print("END-TO-END PIPELINE EVALUATION")
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}  (epoch {ckpt_epoch})")
        print(f"Init method: REGRESSION (direct, skip LoFTR)")
        print(f"Regression poses: {regression_npz_path}")
        print(f"{'='*70}")
        print("\n[Phase 1] Loading regression poses (no LoFTR) ...")

        t_loftr_start = time.time()
        for qi, query in enumerate(test_samples):
            ri = reg_name_to_idx.get(query["img_name"])
            if ri is not None:
                reg_w2c = reg_poses[ri].astype(np.float32)
                success = True
            else:
                # Fallback to nearest-neighbor retrieval pose
                nn_idx = _find_k_nearest(query["camera_center"], train_centers, 1)
                reg_w2c = train_samples[nn_idx[0]]["pose_w2c"].astype(np.float32)
                success = False
                n_loftr_failed += 1
                logger.warning("No regression pose for %s, falling back to retrieval",
                               query["img_name"])

            # Oracle nearest for retrieval baseline metric
            nn_idx = _find_k_nearest(query["camera_center"], train_centers, 1)
            retrieval_w2c = train_samples[nn_idx[0]]["pose_w2c"]

            loftr_best.append({
                "pose_w2c": reg_w2c,
                "retrieval_pose_w2c": retrieval_w2c,
                "success": success,
            })
            loftr_candidates.append([])  # empty, not used for regression

        t_loftr = time.time() - t_loftr_start
        logger.info("Regression init loaded: %.1fs, failures: %d/%d",
                     t_loftr, n_loftr_failed, len(test_samples))

    elif init_method == "regression_loftr":
        # ── Regression-guided retrieval + LoFTR ──
        # Use regression-predicted camera centers for neighbour selection
        # (instead of oracle GT centers), then run LoFTR as normal
        assert regression_npz_path is not None, \
            "--regression_npz is required when --init_method=regression_loftr"
        reg_data = np.load(regression_npz_path, allow_pickle=True)
        reg_names = list(reg_data["img_names"])
        reg_centers_all = reg_data["camera_centers"]  # (N, 3)
        reg_name_to_idx = {n: i for i, n in enumerate(reg_names)}

        # Build predicted center array aligned with test_samples
        pred_centers = np.zeros((len(test_samples), 3), dtype=np.float64)
        for qi, query in enumerate(test_samples):
            ri = reg_name_to_idx.get(query["img_name"])
            if ri is not None:
                pred_centers[qi] = reg_centers_all[ri]
            else:
                pred_centers[qi] = query["camera_center"]  # fallback to GT
                logger.warning("No regression center for %s, using GT", query["img_name"])

        print(f"\n{'='*70}")
        print("END-TO-END PIPELINE EVALUATION")
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}  (epoch {ckpt_epoch})")
        print(f"Init method: REGRESSION-GUIDED LoFTR")
        print(f"Regression poses: {regression_npz_path}")
        print(f"K={args.num_neighbors} neighbors, oi={args.outer_iters}, gru={args.gru_iters}")
        print(f"LoFTR mode: {args.loftr_mode}  Selection: {args.selection}  MS={getattr(args, 'multi_start', 1)}")
        print(f"{'='*70}")
        print("\n[Phase 1] Regression-guided retrieval + LoFTR sparse init ...")

        # Load Gaussians for depth rendering
        from feature_gaussian import HybridGaussianModel as _HGM_loftr
        gaussians_depth = _HGM_loftr(sh_degree=3, latent_dim=0)
        gaussians_depth.load_ply(ply_path, freeze_geometry=True)
        gaussians_depth.active_sh_degree = 3

        from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution
        loftr_hw = _compute_loftr_resolution(orig_hw, args.loftr_long_edge)

        loftr = LoFTRInitializer(
            device=device,
            loftr_long_edge=args.loftr_long_edge,
            confidence_threshold=args.loftr_conf,
            reproj_threshold=getattr(args, "reproj_threshold", 8.0),
            use_magsac=getattr(args, "use_magsac", False),
        )

        t_loftr_start = time.time()
        use_accumulated = getattr(args, "loftr_mode", "individual") == "accumulated"
        with torch.no_grad():
            for qi, query in enumerate(test_samples):
                # KEY DIFFERENCE: use predicted center instead of GT center
                nn_indices = _find_k_nearest(
                    pred_centers[qi], train_centers, args.num_neighbors)
                retrieval_ref = train_samples[nn_indices[0]]

                candidates: List[Dict[str, Any]] = []
                best_cand: Optional[Dict[str, Any]] = None

                for ni in nn_indices:
                    ref = train_samples[ni]
                    ref_depth = _render_depth_for_loftr(
                        gaussians_depth, ref["pose_w2c"], intrinsics,
                        loftr_hw, orig_hw, device,
                    )
                    result = loftr.estimate_pose_direct(
                        query["img_path"], ref["img_path"],
                        ref_depth, ref["pose_w2c"], intrinsics, orig_hw,
                    )
                    cand: Dict[str, Any] = {
                        "pose_w2c": result.pose_w2c if result.success else None,
                        "success": result.success,
                        "num_inliers": result.num_inliers,
                        "ref_idx": ni,
                        "original_ci": len(candidates),
                    }
                    candidates.append(cand)
                    if result.success and (
                        best_cand is None
                        or result.num_inliers > best_cand["num_inliers"]
                    ):
                        best_cand = cand

                loftr_candidates.append(candidates)

                if best_cand is None:
                    n_loftr_failed += 1
                    loftr_best.append({
                        "pose_w2c": retrieval_ref["pose_w2c"],
                        "retrieval_pose_w2c": retrieval_ref["pose_w2c"],
                        "success": False,
                    })
                else:
                    loftr_best.append({
                        "pose_w2c": best_cand["pose_w2c"],
                        "retrieval_pose_w2c": retrieval_ref["pose_w2c"],
                        "success": True,
                    })

                if (qi + 1) % 20 == 0 or qi == 0:
                    elapsed = time.time() - t_loftr_start
                    logger.info("  Reg+LoFTR [%d/%d]  %.1fs  failed=%d",
                                qi + 1, len(test_samples), elapsed, n_loftr_failed)

        t_loftr = time.time() - t_loftr_start
        logger.info("Reg+LoFTR phase: %.1fs (%.2fs/img)  failures: %d/%d",
                     t_loftr, t_loftr / max(1, len(test_samples)),
                     n_loftr_failed, len(test_samples))
        del loftr, gaussians_depth
        torch.cuda.empty_cache()

    else:
        # ── Default: Oracle retrieval + LoFTR ──────────────────────────
        from feature_gaussian import HybridGaussianModel
        gaussians_depth = HybridGaussianModel(sh_degree=3, latent_dim=0)
        gaussians_depth.load_ply(ply_path, freeze_geometry=True)
        gaussians_depth.active_sh_degree = 3
        logger.info("Depth Gaussians: %d points from %s",
                     gaussians_depth.num_points, ply_path)

        from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution
        loftr_hw = _compute_loftr_resolution(orig_hw, args.loftr_long_edge)
        logger.info("LoFTR resolution: %s", loftr_hw)

        loftr = LoFTRInitializer(
            device=device,
            loftr_long_edge=args.loftr_long_edge,
            confidence_threshold=args.loftr_conf,
            reproj_threshold=getattr(args, "reproj_threshold", 8.0),
            use_magsac=getattr(args, "use_magsac", False),
        )

        rematch_iters = getattr(args, "rematch_iters", 0)

        print(f"\n{'='*70}")
        print("END-TO-END PIPELINE EVALUATION")
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}  (epoch {ckpt_epoch})")
        print(f"K={args.num_neighbors} neighbors, oi={args.outer_iters}, gru={args.gru_iters}")
        print(f"LoFTR mode: {args.loftr_mode}  Selection: {args.selection}  MS={getattr(args, 'multi_start', 1)}")
        print(f"{'='*70}")
        print("\n[Phase 1] LoFTR sparse initialisation ...")

        t_loftr_start = time.time()
        use_accumulated = getattr(args, "loftr_mode", "individual") == "accumulated"
        with torch.no_grad():
            for qi, query in enumerate(test_samples):
                nn_indices = _find_k_nearest(
                    query["camera_center"], train_centers, args.num_neighbors)
                retrieval_ref = train_samples[nn_indices[0]]

                candidates: List[Dict[str, Any]] = []
                best_cand: Optional[Dict[str, Any]] = None

                # Collect per-reference individual results
                ref_entries_for_accum = []
                for ni in nn_indices:
                    ref = train_samples[ni]
                    ref_depth = _render_depth_for_loftr(
                        gaussians_depth, ref["pose_w2c"], intrinsics,
                        loftr_hw, orig_hw, device,
                    )
                    result = loftr.estimate_pose_direct(
                        query["img_path"], ref["img_path"],
                        ref_depth, ref["pose_w2c"], intrinsics, orig_hw,
                    )
                    cand: Dict[str, Any] = {
                        "pose_w2c": result.pose_w2c if result.success else None,
                        "success": result.success,
                        "num_inliers": result.num_inliers,
                        "ref_idx": ni,
                        "original_ci": len(candidates),
                    }
                    candidates.append(cand)
                    if result.success and (
                        best_cand is None
                        or result.num_inliers > best_cand["num_inliers"]
                    ):
                        best_cand = cand

                    if use_accumulated:
                        ref_entries_for_accum.append({
                            "img_path": ref["img_path"],
                            "depth": ref_depth,
                            "pose_w2c": ref["pose_w2c"],
                        })

                loftr_candidates.append(candidates)

                # Accumulated PnP: combine all matches from K refs into one PnP
                accum_result = None
                accum_corrs = None
                if use_accumulated and ref_entries_for_accum:
                    query_bgr = cv2.imread(str(query["img_path"]))
                    if query_bgr is not None:
                        query_rgb_img = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)

                        # If rematch is enabled, extract correspondences separately
                        # so we can cache them for the rematch phase.
                        if rematch_iters > 0:
                            from pose_refine.sparse_init import _solve_pnp, _compute_loftr_resolution, _scale_intrinsics
                            all_pts_3d_list = []
                            all_pts_2d_list = []
                            for entry in ref_entries_for_accum:
                                if "rgb" in entry:
                                    ref_rgb = entry["rgb"]
                                elif "img_path" in entry:
                                    ref_bgr = cv2.imread(str(entry["img_path"]))
                                    if ref_bgr is None:
                                        continue
                                    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                                else:
                                    continue
                                pts_3d, pts_2d, conf = loftr.extract_correspondences(
                                    query_rgb_img, ref_rgb,
                                    entry["depth"], entry["pose_w2c"],
                                    intrinsics, orig_hw,
                                )
                                if len(pts_3d) > 0:
                                    all_pts_3d_list.append(pts_3d)
                                    all_pts_2d_list.append(pts_2d)

                            if all_pts_3d_list:
                                pts_3d_all = np.concatenate(all_pts_3d_list, axis=0)
                                pts_2d_all = np.concatenate(all_pts_2d_list, axis=0)
                                accum_corrs = (pts_3d_all, pts_2d_all)
                                loftr_hw_local = _compute_loftr_resolution(orig_hw, loftr.loftr_long_edge)
                                intr_loftr = _scale_intrinsics(intrinsics, orig_hw, loftr_hw_local)
                                pose_w2c_acc, n_inliers = _solve_pnp(
                                    pts_3d_all, pts_2d_all, intr_loftr,
                                    reproj_threshold=args.reproj_threshold,
                                )
                                if pose_w2c_acc is not None:
                                    from pose_refine.sparse_init import LoFTRResult
                                    accum_result = LoFTRResult()
                                    accum_result.success = True
                                    accum_result.pose_w2c = pose_w2c_acc.astype(np.float32)
                                    accum_result.num_inliers = n_inliers
                                    accum_result.extra["total_correspondences"] = len(pts_3d_all)
                        else:
                            accum_result = loftr.estimate_pose_accumulated(
                                query_rgb_img,
                                ref_entries_for_accum,
                                intrinsics,
                                orig_hw,
                            )

                # Best LoFTR result: prefer accumulated if available and successful
                if accum_result is not None and accum_result.success:
                    best_entry = {
                        "pose_w2c": accum_result.pose_w2c,
                        "retrieval_pose_w2c": retrieval_ref["pose_w2c"],
                        "success": True,
                        "accumulated": True,
                        "accum_inliers": accum_result.num_inliers,
                        "accum_correspondences": accum_result.extra.get("total_correspondences", 0),
                    }
                    if accum_corrs is not None:
                        best_entry["_accum_corrs_original"] = accum_corrs
                        best_entry["_accum_corrs_active"] = accum_corrs
                    loftr_best.append(best_entry)
                elif best_cand is None:
                    n_loftr_failed += 1
                    loftr_best.append({
                        "pose_w2c": retrieval_ref["pose_w2c"],
                        "retrieval_pose_w2c": retrieval_ref["pose_w2c"],
                        "success": False,
                    })
                else:
                    loftr_best.append({
                        "pose_w2c": best_cand["pose_w2c"],
                        "retrieval_pose_w2c": retrieval_ref["pose_w2c"],
                        "success": True,
                    })

                if (qi + 1) % 20 == 0 or qi == 0:
                    elapsed = time.time() - t_loftr_start
                    logger.info("  LoFTR [%d/%d]  %.1fs  failed=%d",
                                qi + 1, len(test_samples), elapsed, n_loftr_failed)

        t_loftr = time.time() - t_loftr_start
        logger.info("LoFTR phase: %.1fs (%.2fs/img)  failures: %d/%d",
                     t_loftr, t_loftr / max(1, len(test_samples)),
                     n_loftr_failed, len(test_samples))

        # ══════════════════════════════════════════════════════════════════
        #  Phase 1b: Iterative render-and-rematch (optional)
        # ══════════════════════════════════════════════════════════════════
        if rematch_iters > 0 and use_accumulated:
            from pose_refine.sparse_init import _solve_pnp, _compute_loftr_resolution, _scale_intrinsics
            print(f"\n[Phase 1b] Iterative render-and-rematch ({rematch_iters} iters) ...")
            t_rematch_start = time.time()
            intr_loftr_rm = _scale_intrinsics(intrinsics, orig_hw, loftr_hw)

            for rm_iter in range(rematch_iters):
                n_improved = 0
                for qi, (entry, query) in enumerate(zip(loftr_best, test_samples)):
                    if not entry.get("success", False):
                        continue
                    est_pose = entry["pose_w2c"]

                    # Render RGB+depth at estimated pose
                    rendered_rgb, rendered_depth = _render_rgbd_for_loftr(
                        gaussians_depth, est_pose, intrinsics,
                        loftr_hw, orig_hw, device,
                    )

                    # Read query image (reuse if available)
                    query_bgr = cv2.imread(str(query["img_path"]))
                    if query_bgr is None:
                        continue
                    query_rgb_img = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)

                    # Extract new correspondences from rendered view
                    new_pts_3d, new_pts_2d, new_conf = loftr.extract_correspondences(
                        query_rgb_img, rendered_rgb,
                        rendered_depth, est_pose,
                        intrinsics, orig_hw,
                    )

                    if len(new_pts_3d) < 10:
                        continue

                    rematch_cumulative = getattr(args, "rematch_cumulative", False)
                    base_corrs = entry.get("_accum_corrs_active", None)
                    if base_corrs is None:
                        base_corrs = entry.get("_accum_corrs_original", None)
                    if base_corrs is None:
                        continue  # skip entries without original correspondences

                    # Default behavior reuses only the phase-1 correspondences plus
                    # the current render matches. Optional cumulative mode keeps
                    # accepted rematch correspondences across iterations.
                    all_pts_3d = np.concatenate([base_corrs[0], new_pts_3d], axis=0)
                    all_pts_2d = np.concatenate([base_corrs[1], new_pts_2d], axis=0)

                    # Re-solve PnP with accumulated correspondences
                    new_pose, new_inliers = _solve_pnp(
                        all_pts_3d, all_pts_2d, intr_loftr_rm,
                        reproj_threshold=args.reproj_threshold,
                    )

                    # Only accept if the new solution has MORE inliers
                    old_inliers = entry.get("accum_inliers", 0)
                    if new_pose is not None and new_inliers > old_inliers:
                        entry["pose_w2c"] = new_pose.astype(np.float32)
                        entry["accum_inliers"] = new_inliers
                        entry["rematch_iter"] = rm_iter + 1
                        entry["rematch_new_matches"] = len(new_pts_3d)
                        if rematch_cumulative:
                            entry["_accum_corrs_active"] = (all_pts_3d, all_pts_2d)
                        n_improved += 1

                    if (qi + 1) % 40 == 0:
                        logger.info("  Rematch iter %d [%d/%d]  improved=%d",
                                    rm_iter + 1, qi + 1, len(test_samples), n_improved)

                logger.info("Rematch iter %d: %d/%d images updated",
                            rm_iter + 1, n_improved, len(test_samples))

            t_rematch = time.time() - t_rematch_start
            logger.info("Rematch phase: %.1fs (%.2fs/img)",
                        t_rematch, t_rematch / max(1, len(test_samples)))

        # Free LoFTR and depth-only Gaussians to reclaim GPU memory
        del loftr, gaussians_depth
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: Multi-start neural refinement
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Phase 2] Multi-start neural refinement ...")
    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = args.gru_iters
    outer_iters = args.outer_iters

    retrieval_rot: List[float] = []
    retrieval_trans: List[float] = []
    loftr_rot: List[float] = []
    loftr_trans: List[float] = []
    refined_rot: List[float] = []
    refined_trans: List[float] = []
    refined_poses: List[np.ndarray] = []  # Store for potential stage-2
    per_image_records: List[Dict[str, Any]] = []

    t_refine_start = time.time()

    with torch.no_grad():
        for qi, query in enumerate(test_samples):
            gt_w2c_np = query["pose_w2c"]

            # ── Stage 0: Retrieval error ──────────────────────────────
            retr_w2c = loftr_best[qi]["retrieval_pose_w2c"]
            r_rot, r_trans = _pose_errors(retr_w2c, gt_w2c_np)
            retrieval_rot.append(r_rot)
            retrieval_trans.append(r_trans)

            # ── Stage 1: Best LoFTR init error ────────────────────────
            loftr_w2c = loftr_best[qi]["pose_w2c"]
            l_rot, l_trans = _pose_errors(loftr_w2c, gt_w2c_np)
            loftr_rot.append(l_rot)
            loftr_trans.append(l_trans)

            # ── Stage 2+3: Refine each candidate, select best ────────
            ds_idx = name_to_ds_idx.get(query["img_name"])
            if ds_idx is not None:
                sample = val_ds[ds_idx]
                query_fine = sample["query_fine"].unsqueeze(0).to(device)
                query_coarse = sample.get("query_coarse")
                if query_coarse is not None and use_coarse:
                    query_coarse = query_coarse.unsqueeze(0).to(device)
                else:
                    query_coarse = None
            else:
                query_fine = None
                query_coarse = None

            selection_mode = getattr(args, "selection", "residual")

            rerank_mode = getattr(args, "feature_rerank", "off")
            rerank_use_fsm = bool(getattr(args, "feature_rerank_use_fsm", False))
            rerank_top_m = max(1, int(getattr(args, "feature_rerank_top_m", 1)))
            rerank_applied = (
                rerank_mode != "off"
                and query_fine is not None
                and selection_mode != "inlier"
            )
            rerank_candidates_info: List[Dict[str, Any]] = []
            candidate_entries: List[Dict[str, Any]] = [
                {
                    "cand": cand,
                    "original_ci": int(cand.get("original_ci", ci)),
                    "slot_after_rerank": ci,
                }
                for ci, cand in enumerate(loftr_candidates[qi])
            ]

            if rerank_applied:
                for list_idx, entry in enumerate(candidate_entries):
                    cand = entry["cand"]
                    if cand["success"] and cand["pose_w2c"] is not None:
                        score_pose = cand["pose_w2c"]
                    else:
                        score_pose = train_samples[cand["ref_idx"]]["pose_w2c"]
                    pose_t = torch.from_numpy(score_pose.astype(np.float32)).to(device).unsqueeze(0)
                    render_bundle = _render_feature_bundle(pose_t)
                    total_score, fine_score, coarse_score = _feature_rerank_score(
                        query_fine,
                        query_coarse,
                        render_bundle,
                        use_fsm_conf=rerank_use_fsm,
                    )
                    rerank_candidates_info.append({
                        "ci": int(entry["original_ci"]),
                        "ref_idx": int(cand["ref_idx"]),
                        "loftr_success": bool(cand["success"]),
                        "num_inliers": int(cand["num_inliers"]),
                        "total_score": float(total_score),
                        "fine_score": float(fine_score),
                        "coarse_score": float(coarse_score),
                        "_list_idx": list_idx,
                    })

                rerank_candidates_info.sort(key=lambda item: item["total_score"])
                candidate_entries = [
                    candidate_entries[item["_list_idx"]]
                    for item in rerank_candidates_info
                ]
                for slot, entry in enumerate(candidate_entries):
                    entry["slot_after_rerank"] = slot
                for item in rerank_candidates_info:
                    item.pop("_list_idx", None)

            if rerank_applied:
                candidates_to_refine = candidate_entries[:min(rerank_top_m, len(candidate_entries))]
            else:
                candidates_to_refine = candidate_entries

            selected_original_ci = -1
            selected_slot_after_rerank = -1
            best_feature_residual = float("inf")
            best_selection_fine = float("inf")
            best_selection_coarse = 0.0
            rerank_top_original_ci = (
                int(rerank_candidates_info[0]["ci"])
                if rerank_candidates_info else -1
            )

            # For 'inlier' selection: only refine the best LoFTR candidate
            if selection_mode == "inlier":
                # Use accumulated or best-by-inliers pose
                init_w2c = loftr_best[qi]["pose_w2c"]
                best_nb_idx = 0
                best_residual = 0.0
                best_feature_residual = 0.0
                best_selection_fine = 0.0
                best_selection_coarse = 0.0
                ms_count = getattr(args, "multi_start", 1)
                ms_noise_deg = getattr(args, "ms_noise_deg", 0.5)
                ms_noise_m = getattr(args, "ms_noise_m", 0.05)
                ms_selection = getattr(args, "ms_selection", "consensus")

                if query_fine is not None and outer_iters > 0:
                    rng = np.random.RandomState(qi)
                    all_ms_poses = []

                    for si in range(ms_count):
                        if si == 0:
                            start_w2c = init_w2c  # first start is unperturbed
                        else:
                            start_w2c = _perturb_pose(
                                init_w2c, ms_noise_deg, ms_noise_m, rng)

                        pose_cur = torch.from_numpy(
                            start_w2c.astype(np.float32)).to(device).unsqueeze(0)
                        for _ in range(outer_iters):
                            ref_fine, depth_r = render_batch(
                                gaussians_dcff, dcff_renderer, feat_sharp,
                                pose_cur, K, render_h, render_w,
                            )
                            with autocast(enabled=True):
                                pred = model(
                                    query_fine, ref_fine, depth_r,
                                    intrinsics=render_intr,
                                    query_coarse=query_coarse,
                                )
                            with torch.cuda.amp.autocast(enabled=False):
                                if "delta_xi" in pred:
                                    T_delta = se3_exp(pred["delta_xi"].float())
                                    pose_cur = torch.bmm(T_delta, pose_cur.float())

                        pose_np = pose_cur.squeeze(0).cpu().numpy()
                        # Compute feature residual for this start
                        ref_fine_ms, _ = render_batch(
                            gaussians_dcff, dcff_renderer, feat_sharp,
                            pose_cur, K, render_h, render_w,
                        )
                        res_ms = float(
                            (query_fine - ref_fine_ms).pow(2).mean().item())
                        all_ms_poses.append((pose_np, res_ms))

                    if len(all_ms_poses) == 1:
                        best_refined_pose = all_ms_poses[0][0]
                        best_residual = all_ms_poses[0][1]
                        best_feature_residual = all_ms_poses[0][1]
                        best_selection_fine = all_ms_poses[0][1]
                    elif ms_selection == "consensus":
                        trans_vecs = np.array([p[:3, 3] for p, _ in all_ms_poses])
                        centroid = np.median(trans_vecs, axis=0)
                        dists = np.linalg.norm(trans_vecs - centroid, axis=1)
                        best_si = int(np.argmin(dists))
                        best_refined_pose, best_residual = all_ms_poses[best_si]
                        best_feature_residual = all_ms_poses[best_si][1]
                        best_selection_fine = all_ms_poses[best_si][1]
                    else:  # residual
                        best_si = int(np.argmin([r for _, r in all_ms_poses]))
                        best_refined_pose, best_residual = all_ms_poses[best_si]
                        best_feature_residual = all_ms_poses[best_si][1]
                        best_selection_fine = all_ms_poses[best_si][1]
                else:
                    best_refined_pose = init_w2c
            else:
                # Refine all candidates, or only top-M after feature reranking.
                all_refined = []  # (pose_np, selection_score, fine_residual, inlier_count, slot, original_ci, score_fine, score_coarse)
                for entry in candidates_to_refine:
                    cand = entry["cand"]
                    slot_after_rerank = int(entry["slot_after_rerank"])
                    original_ci = int(entry["original_ci"])
                    if cand["success"] and cand["pose_w2c"] is not None:
                        init_w2c_ci = cand["pose_w2c"]
                    else:
                        init_w2c_ci = train_samples[cand["ref_idx"]]["pose_w2c"]

                    if query_fine is None:
                        all_refined.append((
                            init_w2c_ci,
                            1e6,
                            1e6,
                            cand["num_inliers"],
                            slot_after_rerank,
                            original_ci,
                            1e6,
                            0.0,
                        ))
                        continue

                    pose_cur = torch.from_numpy(
                        init_w2c_ci.astype(np.float32)).to(device).unsqueeze(0)

                    for _ in range(outer_iters):
                        ref_fine, depth_r = render_batch(
                            gaussians_dcff, dcff_renderer, feat_sharp,
                            pose_cur, K, render_h, render_w,
                        )
                        with autocast(enabled=True):
                            pred = model(
                                query_fine, ref_fine, depth_r,
                                intrinsics=render_intr,
                                query_coarse=query_coarse,
                            )
                        with torch.cuda.amp.autocast(enabled=False):
                            if "delta_xi" in pred:
                                T_delta = se3_exp(pred["delta_xi"].float())
                                pose_cur = torch.bmm(T_delta, pose_cur.float())

                    # Final selection score. With feature rerank enabled, keep coarse/FSM
                    # in the decision path instead of falling back to fine-only residual.
                    if rerank_applied:
                        final_bundle = _render_feature_bundle(pose_cur)
                        ref_fine_final = final_bundle["fine_features"]
                        fine_residual = float((query_fine - ref_fine_final).pow(2).mean().item())
                        selection_total, selection_fine, selection_coarse = _feature_rerank_score(
                            query_fine,
                            query_coarse,
                            final_bundle,
                            use_fsm_conf=rerank_use_fsm,
                        )
                    else:
                        ref_fine_final, _ = render_batch(
                            gaussians_dcff, dcff_renderer, feat_sharp,
                            pose_cur, K, render_h, render_w,
                        )
                        fine_residual = float(
                            (query_fine - ref_fine_final).pow(2).mean().item())
                        selection_total = fine_residual
                        selection_fine = fine_residual
                        selection_coarse = 0.0

                    all_refined.append((
                        pose_cur.squeeze(0).cpu().numpy(),
                        selection_total,
                        fine_residual,
                        cand["num_inliers"],
                        slot_after_rerank,
                        original_ci,
                        selection_fine,
                        selection_coarse,
                    ))

                if not all_refined:
                    best_refined_pose = retr_w2c
                    best_nb_idx = -1
                    best_residual = float("inf")
                    best_feature_residual = float("inf")
                    best_selection_fine = float("inf")
                    best_selection_coarse = 0.0
                    selected_original_ci = -1
                    selected_slot_after_rerank = -1
                elif selection_mode == "consensus":
                    # Consensus: pick pose closest to translation centroid
                    trans_vecs = np.array([p[:3, 3] for p, _, _, _, _, _, _, _ in all_refined])
                    centroid = np.median(trans_vecs, axis=0)
                    dists = np.linalg.norm(trans_vecs - centroid, axis=1)
                    best_idx = int(np.argmin(dists))
                    (
                        best_refined_pose,
                        best_residual,
                        best_feature_residual,
                        _,
                        selected_slot_after_rerank,
                        selected_original_ci,
                        best_selection_fine,
                        best_selection_coarse,
                    ) = all_refined[best_idx]
                    best_nb_idx = selected_slot_after_rerank
                else:  # "residual"
                    best_idx = int(np.argmin([r for _, r, _, _, _, _, _, _ in all_refined]))
                    (
                        best_refined_pose,
                        best_residual,
                        best_feature_residual,
                        _,
                        selected_slot_after_rerank,
                        selected_original_ci,
                        best_selection_fine,
                        best_selection_coarse,
                    ) = all_refined[best_idx]
                    best_nb_idx = selected_slot_after_rerank

            # Fallback: nearest-neighbour if nothing survived
            if best_refined_pose is None:
                best_refined_pose = retr_w2c

            rf_rot, rf_trans = _pose_errors(best_refined_pose, gt_w2c_np)
            refined_rot.append(rf_rot)
            refined_trans.append(rf_trans)
            refined_poses.append(best_refined_pose)

            per_image_records.append({
                "image": query["img_name"],
                "retrieval_rot_deg": round(r_rot, 3),
                "retrieval_trans_m": round(r_trans, 4),
                "loftr_rot_deg": round(l_rot, 3),
                "loftr_trans_m": round(l_trans, 4),
                "refined_rot_deg": round(rf_rot, 3),
                "refined_trans_m": round(rf_trans, 4),
                "loftr_success": loftr_best[qi]["success"],
                "selected_nb": best_nb_idx,
                "selected_slot_after_rerank": selected_slot_after_rerank,
                "selected_ci_original": selected_original_ci,
                "feature_residual": round(best_feature_residual, 6),
                "selection_score_total": round(best_residual, 6),
                "selection_score_fine": round(best_selection_fine, 6),
                "selection_score_coarse": round(best_selection_coarse, 6),
                "rerank_applied": rerank_applied,
                "rerank_mode": rerank_mode,
                "rerank_used_coarse": query_coarse is not None,
                "rerank_used_fsm_conf": rerank_use_fsm,
                "rerank_refine_top_m": (min(rerank_top_m, len(candidate_entries)) if rerank_applied else len(candidate_entries)),
                "rerank_top_original_ci": rerank_top_original_ci,
                "num_candidates_total": len(candidate_entries),
                "num_candidates_refined": len(candidates_to_refine),
                "rerank_candidates": rerank_candidates_info,
            })

            # Per-image progress line
            print(f"  [{qi+1:>4d}/{len(test_samples)}]  {query['img_name']:<30s}  "
                  f"retrieval={r_rot:.2f}\u00b0/{r_trans:.2f}m  "
                  f"loftr={l_rot:.2f}\u00b0/{l_trans:.2f}m  "
                  f"refined={rf_rot:.2f}\u00b0/{rf_trans:.2f}m")

    model.gru_iters = original_gru
    t_refine = time.time() - t_refine_start

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 3 (optional): Stage-2 precision refinement
    # ══════════════════════════════════════════════════════════════════════
    stage2_rot: Optional[List[float]] = None
    stage2_trans: Optional[List[float]] = None
    t_stage2 = 0.0

    if getattr(args, "stage2_config", None) and getattr(args, "stage2_checkpoint", None):
        print("\n[Phase 3] Stage-2 precision refinement ...")
        stage2_config = load_mainline_config(args.stage2_config)
        s2_model, s2_epoch = load_model(stage2_config, args.stage2_checkpoint, device)
        s2_ckpt = load_concat_pose_checkpoint(args.stage2_checkpoint, device)
        restored_stage2_map = apply_localization_map_state(
            dcff_renderer,
            feat_sharp,
            s2_ckpt,
            printer=logger.info,
        )
        if restored_stage2_map:
            logger.info("Restored stage-2 map state: %s", ", ".join(restored_stage2_map))
        s2_model.eval()
        s2_model.BASE_INTRINSICS = model.BASE_INTRINSICS
        s2_model.IMG_HW = model.IMG_HW
        s2_render_intr = s2_model._scale_intrinsics(render_h, render_w)

        s2_use_coarse = stage2_config.get("model", {}).get("use_coarse", True)
        s2_oi = getattr(args, "stage2_outer_iters", 5)
        s2_gi = getattr(args, "stage2_gru_iters", 6)
        s2_model.gru_iters = s2_gi

        stage2_rot = []
        stage2_trans = []
        t_s2_start = time.time()

        with torch.no_grad():
            for qi, query in enumerate(test_samples):
                gt_w2c_np = query["pose_w2c"]
                init_w2c = refined_poses[qi]

                ds_idx = name_to_ds_idx.get(query["img_name"])
                if ds_idx is not None:
                    sample = val_ds[ds_idx]
                    query_fine = sample["query_fine"].unsqueeze(0).to(device)
                    query_coarse = sample.get("query_coarse")
                    if query_coarse is not None and s2_use_coarse:
                        query_coarse = query_coarse.unsqueeze(0).to(device)
                    else:
                        query_coarse = None
                else:
                    query_fine = None
                    query_coarse = None

                if query_fine is None:
                    s2_rot, s2_trans_v = _pose_errors(init_w2c, gt_w2c_np)
                else:
                    pose_cur = torch.from_numpy(
                        init_w2c.astype(np.float32)).to(device).unsqueeze(0)
                    for _ in range(s2_oi):
                        ref_fine, depth_r = render_batch(
                            gaussians_dcff, dcff_renderer, feat_sharp,
                            pose_cur, K, render_h, render_w,
                        )
                        with autocast(enabled=True):
                            pred = s2_model(
                                query_fine, ref_fine, depth_r,
                                intrinsics=s2_render_intr,
                                query_coarse=query_coarse,
                            )
                        with torch.cuda.amp.autocast(enabled=False):
                            if "delta_xi" in pred:
                                T_delta = se3_exp(pred["delta_xi"].float())
                                pose_cur = torch.bmm(T_delta, pose_cur.float())
                    s2_final = pose_cur.squeeze(0).cpu().numpy()
                    s2_rot, s2_trans_v = _pose_errors(s2_final, gt_w2c_np)
                    per_image_records[qi]["stage2_rot_deg"] = round(s2_rot, 3)
                    per_image_records[qi]["stage2_trans_m"] = round(s2_trans_v, 4)

                stage2_rot.append(s2_rot)
                stage2_trans.append(s2_trans_v)

                if (qi + 1) % 10 == 0 or qi + 1 == len(test_samples):
                    s2r = np.array(stage2_rot)
                    s2t = np.array(stage2_trans) * 1000
                    logger.info(
                        "  [%3d/%d]  s2_rot=%.2f°  s2_trans=%.0fmm",
                        qi + 1, len(test_samples),
                        np.median(s2r), np.median(s2t),
                    )

        t_stage2 = time.time() - t_s2_start
        print(f"Stage-2 done in {t_stage2:.1f}s (model epoch {s2_epoch})")
        del s2_model
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    #  Results summary
    # ══════════════════════════════════════════════════════════════════════
    total_elapsed = t_loftr + t_refine + t_stage2

    ret_r = np.array(retrieval_rot)
    ret_t = np.array(retrieval_trans)
    lof_r = np.array(loftr_rot)
    lof_t = np.array(loftr_trans)
    ref_r = np.array(refined_rot)
    ref_t = np.array(refined_trans)

    # Convert to mm for display
    ret_t_mm = ret_t * 1000
    lof_t_mm = lof_t * 1000
    ref_t_mm = ref_t * 1000

    def pct_under(r, t, r_thr, t_thr):
        return float(np.mean((r < r_thr) & (t < t_thr))) * 100.0

    sep = "=" * 70
    print(f"\n{sep}")
    print("RESULTS SUMMARY")
    print(sep)
    print(f"Time: LoFTR={t_loftr:.1f}s  Refine={t_refine:.1f}s  "
          f"Total={total_elapsed:.1f}s  ({total_elapsed/len(test_samples):.1f}s/img)")
    print(f"LoFTR failures: {n_loftr_failed}/{len(test_samples)}")

    deg = "\u00b0"
    header = ("\n{:<20s} {:>12s} {:>16s} {:>10s} {:>12s}".format(
        "Stage", "Rot Med (" + deg + ")", "Trans Med (mm)",
        "<1" + deg + "/50mm", "<0.5" + deg + "/30mm"))
    print(header)
    print("-" * 72)
    for label, rots, trans_mm in [
        ("Retrieval", ret_r, ret_t_mm),
        ("LoFTR init", lof_r, lof_t_mm),
        ("Refined (S1)", ref_r, ref_t_mm),
    ]:
        j1 = pct_under(rots, trans_mm, 1.0, 50.0)
        j05 = pct_under(rots, trans_mm, 0.5, 30.0)
        print(f"{label:<20s} {np.median(rots):>11.2f} "
              f"{np.median(trans_mm):>15.1f} "
              f"{j1:>9.1f}% {j05:>11.1f}%")

    # Stage-2 results (the "final" output for SOTA comparison)
    if stage2_rot is not None:
        s2_r = np.array(stage2_rot)
        s2_t = np.array(stage2_trans) * 1000
        j1 = pct_under(s2_r, s2_t, 1.0, 50.0)
        j05 = pct_under(s2_r, s2_t, 0.5, 30.0)
        print(f"{'Refined (S2)' :<20s} {np.median(s2_r):>11.2f} "
              f"{np.median(s2_t):>15.1f} "
              f"{j1:>9.1f}% {j05:>11.1f}%")
        # Use stage-2 as final for percentiles & SOTA comparison
        final_r, final_t_mm = s2_r, s2_t
        final_label = "S2"
    else:
        final_r, final_t_mm = ref_r, ref_t_mm
        final_label = "S1"

    print(f"\nPercentiles ({final_label} refined):")
    for p in [25, 50, 75, 90, 95]:
        rp = np.percentile(final_r, p)
        tp = np.percentile(final_t_mm, p)
        print("  p%d: rot=%.2f%s  trans=%.0fmm" % (p, rp, deg, tp))

    print(f"\nThreshold breakdown:")
    if stage2_rot is not None:
        s2_r_arr = np.array(stage2_rot)
        s2_t_mm_arr = np.array(stage2_trans) * 1000
        thresh_header = f"  {'Threshold':<25s} {'Retrieval':>10s} {'LoFTR':>10s} {'S1':>10s} {'S2':>10s}"
        print(thresh_header)
        print("  " + "-" * 67)
        for rdeg, tmm in [(1.0, 50.0), (2.0, 100.0), (5.0, 250.0), (10.0, 500.0)]:
            p_ret = pct_under(ret_r, ret_t_mm, rdeg, tmm)
            p_lof = pct_under(lof_r, lof_t_mm, rdeg, tmm)
            p_ref = pct_under(ref_r, ref_t_mm, rdeg, tmm)
            p_s2 = pct_under(s2_r_arr, s2_t_mm_arr, rdeg, tmm)
            thr_label = "< %s%s / %dmm" % (str(rdeg), deg, int(tmm))
            print("  {:<25s} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
                thr_label, p_ret, p_lof, p_ref, p_s2))
    else:
        thresh_header = f"  {'Threshold':<25s} {'Retrieval':>10s} {'LoFTR':>10s} {'Refined':>10s}"
        print(thresh_header)
        print("  " + "-" * 55)
        for rdeg, tmm in [(1.0, 50.0), (2.0, 100.0), (5.0, 250.0), (10.0, 500.0)]:
            p_ret = pct_under(ret_r, ret_t_mm, rdeg, tmm)
            p_lof = pct_under(lof_r, lof_t_mm, rdeg, tmm)
            p_ref = pct_under(ref_r, ref_t_mm, rdeg, tmm)
            thr_label = "< %s%s / %dmm" % (str(rdeg), deg, int(tmm))
            print("  {:<25s} {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
                thr_label, p_ret, p_lof, p_ref))

    print(f"\nSOTA comparison:")
    print("  GSFFs:  0.32%s / 183mm" % deg)
    print("  Ours:   %.2f%s / %.0fmm" % (np.median(final_r), deg, np.median(final_t_mm)))
    print(sep)

    # ══════════════════════════════════════════════════════════════════════
    #  Save JSON results
    # ══════════════════════════════════════════════════════════════════════
    out_dir = args.output_dir or os.path.join(_PROJ_ROOT, "output", "pipeline_eval")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")

    results_json: Dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "output_dir": out_dir,
        "epoch": str(ckpt_epoch),
        "eval_mode": eval_mode_meta,
        "init_method": getattr(args, "init_method", "loftr"),
        "regression_npz": getattr(args, "regression_npz", None),
        "num_neighbors": args.num_neighbors,
        "outer_iters": args.outer_iters,
        "gru_iters": args.gru_iters,
        "loftr_long_edge": args.loftr_long_edge,
        "loftr_conf": args.loftr_conf,
        "rematch_iters": args.rematch_iters,
        "rematch_cumulative": getattr(args, "rematch_cumulative", False),
        "solver": args.solver,
        "feature_rerank": args.feature_rerank,
        "feature_rerank_use_fsm": bool(args.feature_rerank_use_fsm),
        "feature_rerank_top_m": int(args.feature_rerank_top_m),
        "loftr_failures": n_loftr_failed,
        "total_images": len(test_samples),
        "time_loftr_s": round(t_loftr, 2),
        "time_refine_s": round(t_refine, 2),
        "time_stage2_s": round(t_stage2, 2),
        "time_total_s": round(total_elapsed, 2),
        "retrieval": {
            "rot_median_deg": round(float(np.median(ret_r)), 3),
            "rot_mean_deg": round(float(np.mean(ret_r)), 3),
            "trans_median_mm": round(float(np.median(ret_t_mm)), 1),
            "trans_mean_mm": round(float(np.mean(ret_t_mm)), 1),
        },
        "loftr_init": {
            "rot_median_deg": round(float(np.median(lof_r)), 3),
            "rot_mean_deg": round(float(np.mean(lof_r)), 3),
            "trans_median_mm": round(float(np.median(lof_t_mm)), 1),
            "trans_mean_mm": round(float(np.mean(lof_t_mm)), 1),
        },
        "refined_s1": {
            "rot_median_deg": round(float(np.median(ref_r)), 3),
            "rot_mean_deg": round(float(np.mean(ref_r)), 3),
            "trans_median_mm": round(float(np.median(ref_t_mm)), 1),
            "trans_mean_mm": round(float(np.mean(ref_t_mm)), 1),
            "pct_1deg_50mm": round(pct_under(ref_r, ref_t_mm, 1.0, 50.0), 2),
            "pct_05deg_30mm": round(pct_under(ref_r, ref_t_mm, 0.5, 30.0), 2),
            "pct_5deg_250mm": round(pct_under(ref_r, ref_t_mm, 5.0, 250.0), 2),
            "percentiles": {
                f"p{p}": {
                    "rot_deg": round(float(np.percentile(ref_r, p)), 3),
                    "trans_mm": round(float(np.percentile(ref_t_mm, p)), 1),
                }
                for p in [25, 50, 75, 90, 95, 99]
            },
        },
        "per_image": per_image_records,
    }

    if stage2_rot is not None:
        s2_r_arr = np.array(stage2_rot)
        s2_t_mm_arr = np.array(stage2_trans) * 1000
        results_json["stage2_config"] = args.stage2_config
        results_json["stage2_checkpoint"] = args.stage2_checkpoint
        results_json["stage2_outer_iters"] = args.stage2_outer_iters
        results_json["stage2_gru_iters"] = args.stage2_gru_iters
        results_json["refined_s2"] = {
            "rot_median_deg": round(float(np.median(s2_r_arr)), 3),
            "rot_mean_deg": round(float(np.mean(s2_r_arr)), 3),
            "trans_median_mm": round(float(np.median(s2_t_mm_arr)), 1),
            "trans_mean_mm": round(float(np.mean(s2_t_mm_arr)), 1),
            "pct_1deg_50mm": round(pct_under(s2_r_arr, s2_t_mm_arr, 1.0, 50.0), 2),
            "pct_05deg_30mm": round(pct_under(s2_r_arr, s2_t_mm_arr, 0.5, 30.0), 2),
            "pct_5deg_250mm": round(pct_under(s2_r_arr, s2_t_mm_arr, 5.0, 250.0), 2),
            "percentiles": {
                f"p{p}": {
                    "rot_deg": round(float(np.percentile(s2_r_arr, p)), 3),
                    "trans_mm": round(float(np.percentile(s2_t_mm_arr, p)), 1),
                }
                for p in [25, 50, 75, 90, 95, 99]
            },
        }

    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    summary_lines = [
        f"eval_mode={eval_mode_meta['eval_mode']} deployable={eval_mode_meta['deployable']}",
        f"init={getattr(args, 'init_method', 'loftr')} K={args.num_neighbors}",
        "LoFTR init: {:.3f}deg / {:.1f}mm".format(
            float(np.median(lof_r)),
            float(np.median(lof_t_mm)),
        ),
        "Refined S1: {:.3f}deg / {:.1f}mm".format(
            float(np.median(ref_r)),
            float(np.median(ref_t_mm)),
        ),
    ]
    if stage2_rot is not None:
        summary_lines.append(
            "Refined S2: {:.3f}deg / {:.1f}mm".format(
                float(np.median(s2_r_arr)),
                float(np.median(s2_t_mm_arr)),
            )
        )

    notes = [
        f"config={args.config}",
        f"checkpoint={args.checkpoint}",
        f"selection={args.selection}",
        f"loftr_mode={args.loftr_mode}",
        f"feature_rerank={args.feature_rerank}",
        f"feature_rerank_use_fsm={bool(args.feature_rerank_use_fsm)}",
        f"feature_rerank_top_m={int(args.feature_rerank_top_m)}",
        f"rematch_iters={args.rematch_iters}",
        f"rematch_cumulative={getattr(args, 'rematch_cumulative', False)}",
        f"solver={args.solver}",
        f"eval_mode={eval_mode_meta['eval_mode']}",
        f"uses_gt_query_center={eval_mode_meta['uses_gt_query_center']}",
        f"deployable={eval_mode_meta['deployable']}",
    ]
    artifact_paths = [json_path]
    save_experiment_bundle(
        exp_name=Path(out_dir).name,
        output_dir=out_dir,
        metrics=results_json,
        summary_lines=summary_lines,
        notes=notes,
        artifact_paths=artifact_paths,
        results_json_name="results.json",
        results_text_name="results.txt",
        report_markdown_name="report.md",
        report_text_name="report.txt",
    )
    print(f"\nResults saved to {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: retrieval → LoFTR init → refinement → select",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="YAML config for refinement model")
    parser.add_argument("--checkpoint", required=True,
                        help="Refinement model checkpoint")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index")
    parser.add_argument("--num_neighbors", type=int, default=3,
                        help="K nearest training images for LoFTR")
    parser.add_argument("--outer_iters", type=int, default=5,
                        help="Refinement outer iterations")
    parser.add_argument("--gru_iters", type=int, default=6,
                        help="GRU iterations per outer step")
    parser.add_argument("--loftr_long_edge", type=int, default=840,
                        help="LoFTR resolution long edge")
    parser.add_argument("--loftr_conf", type=float, default=0.3,
                        help="LoFTR confidence threshold")
    parser.add_argument("--reproj_threshold", type=float, default=8.0,
                        help="PnP RANSAC reprojection error threshold (px)")
    parser.add_argument("--use_magsac", action="store_true",
                        help="Use USAC_MAGSAC for PnP instead of RANSAC")
    parser.add_argument("--rematch_iters", type=int, default=0,
                        help="Iterative render-and-rematch iterations after "
                             "initial LoFTR PnP. Renders at estimated pose, "
                             "re-matches with LoFTR, accumulates correspondences.")
    parser.add_argument("--rematch_cumulative", action="store_true",
                        help="Keep accepted rematch correspondences across "
                             "iterations instead of reusing only the original "
                             "phase-1 matches plus the current render matches.")
    parser.add_argument("--solver", type=str, default="default",
                        help="Pose solver: 'default' uses model's delta_xi")
    parser.add_argument("--feature_rerank", type=str, default="off",
                        choices=["off", "coarse_fine"],
                        help="Optional feature-based reranking of LoFTR candidates before refinement")
    parser.add_argument("--feature_rerank_use_fsm", action="store_true",
                        help="Weight feature reranking residuals with FSM spatial confidence when available")
    parser.add_argument("--feature_rerank_top_m", type=int, default=1,
                        help="When feature reranking is enabled, refine only the top-M reranked candidates")
    parser.add_argument("--loftr_mode", type=str, default="individual",
                        choices=["individual", "accumulated"],
                        help="LoFTR matching mode: 'individual' (PnP per ref, "
                             "pick best) or 'accumulated' (combine all matches "
                             "into one PnP)")
    parser.add_argument("--selection", type=str, default="residual",
                        choices=["residual", "consensus", "inlier"],
                        help="Selection strategy: 'residual' (feature MSE), "
                             "'consensus' (translation centroid), "
                             "'inlier' (LoFTR inlier count, refine best only)")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Limit test images (for debugging)")
    # Initialisation method
    parser.add_argument("--eval_mode", type=str, default=None,
                        choices=["oracle", "deploy", "ablation"],
                        help="Explicit evaluation label. Default is oracle for init_method=loftr and deploy otherwise.")
    parser.add_argument("--init_method", type=str, default="loftr",
                        choices=["loftr", "regression", "regression_loftr"],
                        help="How to initialise poses before refinement: "
                             "'loftr' (oracle retrieval + LoFTR, default), "
                             "'regression' (direct MLP regression, skip LoFTR), "
                             "'regression_loftr' (regression-guided retrieval + LoFTR)")
    parser.add_argument("--regression_npz", type=str, default=None,
                        help="Path to .npz with regression-predicted poses "
                             "(required for init_method=regression|regression_loftr). "
                             "Keys: img_names, poses_w2c, camera_centers")
    # Multi-start refinement around LoFTR init
    parser.add_argument("--multi_start", type=int, default=1,
                        help="Number of perturbed starts around LoFTR init "
                             "(1 = no perturbation, just refine init)")
    parser.add_argument("--ms_noise_deg", type=float, default=0.5,
                        help="Rotation noise std for multi-start perturbation")
    parser.add_argument("--ms_noise_m", type=float, default=0.05,
                        help="Translation noise std for multi-start perturbation")
    parser.add_argument("--ms_selection", type=str, default="consensus",
                        choices=["consensus", "residual"],
                        help="Multi-start selection: 'consensus' or 'residual'")
    # Two-stage refinement
    parser.add_argument("--stage2_config", type=str, default=None,
                        help="Optional stage-2 config (precision refinement)")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Optional stage-2 checkpoint")
    parser.add_argument("--stage2_outer_iters", type=int, default=5,
                        help="Stage-2 outer iterations")
    parser.add_argument("--stage2_gru_iters", type=int, default=6,
                        help="Stage-2 GRU iterations")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory for run-specific results.json")
    args = parser.parse_args()

    # Set GPU before importing torch
    _setup_gpu(args.gpu)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(args)


if __name__ == "__main__":
    main()
