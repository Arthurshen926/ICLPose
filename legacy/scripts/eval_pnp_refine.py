#!/usr/bin/env python3
"""Iterative PnP-from-features test-time pose refinement.

Renders DCFF features + depth at current pose, finds dense correspondences
via feature correlation with query, then solves PnP RANSAC to update pose.
No learned model needed — tests DCFF feature quality directly.

Pipeline per iteration:
  1. Render fine features + depth at current pose
  2. Compute dense feature correlation (query → rendered)
  3. Extract 2D-3D correspondences from depth + best-match pixels
  4. PnP RANSAC → relative pose update
  5. Update pose and repeat

Example::
    python scripts/eval_pnp_refine.py \
        --config configs/concat_loc_oh_v22d_stage2_lownoise.yaml \
        --gpu 0 --num_iters 5 --noise_deg 3.0 --noise_trans 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pnp_refine")


def _pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray):
    """Rotation (degrees) and translation (metres) between two w2c matrices."""
    R_pred, t_pred = pred_w2c[:3, :3], pred_w2c[:3, 3]
    R_gt, t_gt = gt_w2c[:3, :3], gt_w2c[:3, 3]
    c_pred = -R_pred.T @ t_pred
    c_gt = -R_gt.T @ t_gt
    trans_err = float(np.linalg.norm(c_pred - c_gt))
    dR = R_pred @ R_gt.T
    cos_angle = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.degrees(np.arccos(cos_angle)))
    return rot_err, trans_err


@torch.no_grad()
def render_features_and_depth(gaussians, dcff_renderer, feat_sharp,
                               pose_w2c, K, render_h, render_w):
    """Render fine features + depth at given pose."""
    viewmat = pose_w2c.float()
    if viewmat.dim() == 2:
        viewmat = viewmat.unsqueeze(0)
    result = dcff_renderer(
        gaussians, viewmat=viewmat.squeeze(0), K=K,
        width=render_w, height=render_h, render_coarse=False,
    )
    fine_feat = result["fine_features"].float()
    depth = result["depth"].float()
    alpha = result.get("alpha")
    fine_feat = feat_sharp(
        fine_feat,
        depth=depth if depth is not None else None,
        alpha=alpha.float() if alpha is not None else None,
    )
    return fine_feat, depth, alpha


@torch.no_grad()
def dense_feature_match(query_feat, rendered_feat, alpha=None,
                         temperature=0.1, top_k=0):
    """Find dense correspondences via feature correlation.

    For each query pixel, find the best matching rendered pixel.

    Args:
        query_feat: [1, C, Hq, Wq]
        rendered_feat: [1, C, Hr, Wr]
        alpha: [1, 1, Hr, Wr] rendered alpha mask
        temperature: softmax temperature for soft matching
        top_k: if > 0, only keep top-k confident matches

    Returns:
        flow: [1, 2, Hr, Wr] flow from rendered to query pixel coords
        confidence: [1, 1, Hr, Wr] matching confidence
    """
    _, C, Hq, Wq = query_feat.shape
    _, _, Hr, Wr = rendered_feat.shape

    # Normalize features
    qf = F.normalize(query_feat, dim=1)  # [1, C, Hq, Wq]
    rf = F.normalize(rendered_feat, dim=1)  # [1, C, Hr, Wr]

    # Reshape for correlation
    rf_flat = rf.reshape(1, C, Hr * Wr)  # [1, C, HrWr]
    qf_flat = qf.reshape(1, C, Hq * Wq)  # [1, C, HqWq]

    # Correlation: [1, HrWr, HqWq]
    corr = torch.bmm(rf_flat.permute(0, 2, 1), qf_flat) / temperature

    # For each rendered pixel, find best matching query pixel
    # Soft argmax for sub-pixel accuracy
    soft_corr = F.softmax(corr, dim=2)  # [1, HrWr, HqWq]

    # Query pixel coordinates
    device = query_feat.device
    qv, qu = torch.meshgrid(
        torch.arange(Hq, device=device, dtype=torch.float32),
        torch.arange(Wq, device=device, dtype=torch.float32),
        indexing='ij',
    )
    qu_flat = qu.reshape(-1)  # [HqWq]
    qv_flat = qv.reshape(-1)

    # Expected query coordinates for each rendered pixel
    matched_u = (soft_corr[0] @ qu_flat).reshape(Hr, Wr)  # [Hr, Wr]
    matched_v = (soft_corr[0] @ qv_flat).reshape(Hr, Wr)

    # Rendered pixel coordinates
    rv, ru = torch.meshgrid(
        torch.arange(Hr, device=device, dtype=torch.float32),
        torch.arange(Wr, device=device, dtype=torch.float32),
        indexing='ij',
    )

    # Flow: rendered pixel → matched query pixel
    flow_u = matched_u - ru  # [Hr, Wr]
    flow_v = matched_v - rv
    flow = torch.stack([flow_u, flow_v], dim=0).unsqueeze(0)  # [1, 2, Hr, Wr]

    # Confidence: max correlation value
    max_corr, _ = soft_corr.max(dim=2)  # [1, HrWr]
    confidence = max_corr.reshape(1, 1, Hr, Wr)

    # Mask by alpha
    if alpha is not None:
        valid = (alpha > 0.5).float()
        flow = flow * valid
        confidence = confidence * valid

    return flow, confidence


def pnp_from_flow(flow, depth, intrinsics, confidence=None,
                   reproj_thresh=8.0, n_iters=2000):
    """Solve PnP from dense flow + depth.

    Args:
        flow: [1, 2, H, W] flow from rendered to query pixels
        depth: [1, 1, H, W] rendered depth
        intrinsics: dict {fx, fy, cx, cy}
        confidence: [1, 1, H, W] optional confidence

    Returns:
        delta_pose: [4, 4] relative pose update (w2c_new = delta @ w2c_old)
        n_inliers: number of RANSAC inliers
    """
    _, _, H, W = flow.shape
    device = flow.device

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    if depth.dim() == 4:
        depth = depth.squeeze(1)  # [1, H, W]

    z = depth[0]  # [H, W]
    valid = z > 0.05

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )

    # 3D points in rendered camera frame
    x3d = (u_coords - cx) * z / fx
    y3d = (v_coords - cy) * z / fy
    pts_3d = torch.stack([x3d, y3d, z], dim=-1)  # [H, W, 3]

    # Target 2D pixels = rendered pixel + flow
    fu = flow[0, 0]  # [H, W]
    fv = flow[0, 1]
    u_target = u_coords + fu
    v_target = v_coords + fv
    pts_2d = torch.stack([u_target, v_target], dim=-1)  # [H, W, 2]

    # Filter
    mask = valid
    if confidence is not None:
        conf = confidence[0, 0]  # [H, W]
        # Adaptive threshold: softmax over HW pixels gives small values
        conf_thresh = max(0.001, float(conf[mask].median()) * 0.5) if mask.any() else 0.001
        mask = mask & (conf > conf_thresh)

    flat_3d = pts_3d[mask].cpu().numpy().astype(np.float64)
    flat_2d = pts_2d[mask].cpu().numpy().astype(np.float64)

    if len(flat_3d) < 10:
        return np.eye(4), 0

    # Subsample if too many
    if len(flat_3d) > 5000:
        idx = np.random.choice(len(flat_3d), 5000, replace=False)
        flat_3d = flat_3d[idx]
        flat_2d = flat_2d[idx]

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            flat_3d, flat_2d, camera_matrix, None,
            iterationsCount=n_iters,
            reprojectionError=reproj_thresh,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < 4:
            return np.eye(4), 0

        # Refine with inliers
        n_inliers = len(inliers)
        if n_inliers >= 6:
            success, rvec, tvec = cv2.solvePnP(
                flat_3d[inliers.flatten()],
                flat_2d[inliers.flatten()],
                camera_matrix, None,
                rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        R_pnp, _ = cv2.Rodrigues(rvec)
        T_pnp = np.eye(4)
        T_pnp[:3, :3] = R_pnp
        T_pnp[:3, 3] = tvec.flatten()
        return T_pnp, n_inliers

    except Exception:
        return np.eye(4), 0


def iterative_pnp_refine(
    query_feat, init_pose, gaussians, dcff_renderer, feat_sharp,
    K, intrinsics, render_h, render_w, num_iters=5,
    temperature=0.1, reproj_thresh=8.0,
):
    """Iteratively refine pose via feature matching + PnP.

    Returns:
        final_pose: [4, 4] numpy w2c
        history: list of (rot_err, trans_err, n_inliers) per iter
    """
    device = query_feat.device
    current_pose = init_pose.clone().to(device)

    history = []
    for it in range(num_iters):
        # Render at current pose
        ref_feat, depth, alpha = render_features_and_depth(
            gaussians, dcff_renderer, feat_sharp,
            current_pose, K, render_h, render_w,
        )

        # Dense feature matching
        flow, confidence = dense_feature_match(
            query_feat, ref_feat, alpha=alpha, temperature=temperature,
        )

        # PnP solve
        delta_T, n_inliers = pnp_from_flow(
            flow, depth, intrinsics, confidence=confidence,
            reproj_thresh=reproj_thresh,
        )

        # Update pose: new_w2c = delta @ current_w2c
        current_pose_np = current_pose.cpu().numpy()
        new_pose = delta_T @ current_pose_np
        current_pose = torch.from_numpy(new_pose.astype(np.float32)).to(device)

        history.append(n_inliers)

    return current_pose.cpu().numpy(), history


def main():
    parser = argparse.ArgumentParser(description="PnP-from-features pose refinement")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_iters", type=int, default=5, help="PnP iterations")
    parser.add_argument("--noise_deg", type=float, default=3.0)
    parser.add_argument("--noise_trans", type=float, default=0.1)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reproj_thresh", type=float, default=8.0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from utils.project_config import load_mainline_config
    from scene_feature_field import build_dcff, intrinsics_to_K

    config = load_mainline_config(args.config)
    logger.info("Building DCFF rendering pipeline...")
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)

    gaussians.set_geometry_trainable(False)
    if hasattr(gaussians, '_latent'):
        gaussians._latent.requires_grad_(False)
    dcff_renderer.eval()
    for p in dcff_renderer.parameters():
        p.requires_grad_(False)
    feat_sharp.eval()
    for p in feat_sharp.parameters():
        p.requires_grad_(False)

    dcff_cfg = config["dcff"]
    render_h = dcff_cfg.get("render_height", 68)
    render_w = dcff_cfg.get("render_width", 120)
    ds_cfg = config["dataset"]

    from data.radio_loc_dataset import RadioLocDataset
    val_ds = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        split="train",
        split_file=ds_cfg.get("test_split"),
        fine_hw=(render_h, render_w),
        coarse_hw=(render_h, render_w),
        noise_rot_deg=args.noise_deg,
        noise_trans_m=args.noise_trans,
        cache_in_memory=True,
    )

    base_intr = val_ds.intrinsics
    orig_h, orig_w = 1080, 1920
    sx, sy = render_w / orig_w, render_h / orig_h
    render_intr = {
        'fx': base_intr['fx'] * sx, 'fy': base_intr['fy'] * sy,
        'cx': base_intr['cx'] * sx, 'cy': base_intr['cy'] * sy,
    }
    K = intrinsics_to_K(render_intr, device)

    logger.info("Render: %d×%d  PnP iters: %d  Temp: %.2f  Seeds: %d  Test: %d",
                render_w, render_h, args.num_iters, args.temperature,
                args.num_seeds, len(val_ds))
    logger.info("Intrinsics (render): fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                render_intr['fx'], render_intr['fy'], render_intr['cx'], render_intr['cy'])

    n_test = len(val_ds) if args.max_test is None else min(args.max_test, len(val_ds))

    all_seed_results = []
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        init_rots, init_trans = [], []
        opt_rots, opt_trans = [], []

        t_start = time.time()
        for i in range(n_test):
            sample = val_ds[i]
            query_fine = sample["query_fine"].unsqueeze(0).to(device)
            noisy_pose = sample["pose_init"].numpy()
            gt_pose = sample["pose_gt"].numpy()

            i_rot, i_trans = _pose_errors(noisy_pose, gt_pose)
            init_rots.append(i_rot)
            init_trans.append(i_trans)

            init_pose_t = torch.from_numpy(noisy_pose.astype(np.float32)).to(device)
            opt_pose, history = iterative_pnp_refine(
                query_fine, init_pose_t,
                gaussians, dcff_renderer, feat_sharp,
                K, render_intr, render_h, render_w,
                num_iters=args.num_iters,
                temperature=args.temperature,
                reproj_thresh=args.reproj_thresh,
            )

            o_rot, o_trans = _pose_errors(opt_pose, gt_pose)
            opt_rots.append(o_rot)
            opt_trans.append(o_trans)

            delta_mm = (o_trans - i_trans) * 1000
            logger.info(
                "  [%3d/%d] seed=%d  init=%.2f°/%.0fmm  opt=%.2f°/%.0fmm  Δ=%.0fmm  inliers=%s",
                i + 1, n_test, seed,
                i_rot, i_trans * 1000,
                o_rot, o_trans * 1000,
                delta_mm,
                [h for h in history],
            )

        elapsed = time.time() - t_start
        ir_med = float(np.median(init_rots))
        it_med = float(np.median(init_trans)) * 1000
        or_med = float(np.median(opt_rots))
        ot_med = float(np.median(opt_trans)) * 1000
        logger.info(
            "[seed %d] %d iters  init=%.2f°/%.0fmm  opt=%.2f°/%.0fmm  (%.1fs)",
            seed, args.num_iters, ir_med, it_med, or_med, ot_med, elapsed,
        )
        all_seed_results.append({
            "seed": seed,
            "init_rot_med": ir_med,
            "init_trans_med": it_med,
            "opt_rot_med": or_med,
            "opt_trans_med": ot_med,
            "elapsed": elapsed,
        })

    # Save results
    out_dir = os.path.join(config.get("output_dir", "output"), "pnp_refine_eval")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"it{args.num_iters}_t{args.temperature}_noise{args.noise_deg}deg_{args.noise_trans}m"
    out_path = os.path.join(out_dir, f"results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": args.config,
            "num_iters": args.num_iters,
            "temperature": args.temperature,
            "reproj_thresh": args.reproj_thresh,
            "noise_deg": args.noise_deg,
            "noise_trans": args.noise_trans,
            "seeds": all_seed_results,
        }, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Summary
    print()
    print("=" * 70)
    print("PNP-FROM-FEATURES REFINEMENT RESULTS")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Noise: {args.noise_deg}° / {args.noise_trans}m")
    print(f"PnP iters: {args.num_iters}  Temp: {args.temperature}")
    print(f"Seeds: {args.num_seeds}")
    print()
    for sr in all_seed_results:
        print(f"  seed {sr['seed']}: init={sr['init_rot_med']:.2f}°/{sr['init_trans_med']:.0f}mm"
              f"  opt={sr['opt_rot_med']:.2f}°/{sr['opt_trans_med']:.0f}mm"
              f"  ({sr['elapsed']:.0f}s)")
    print()
    print(f"  SOTA (GSFFs): 0.32° / 183mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
