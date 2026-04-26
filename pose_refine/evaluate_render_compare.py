#!/usr/bin/env python3
"""Canonical PoseRefine render-and-compare evaluation entrypoint.

Render-and-compare test-time pose optimization.

Optimizes pose directly via gradient descent on feature similarity
between rendered map features and query features. No learned refinement model.
Tests DCFF feature quality independent of ConcatPoseNet.

Example::
    python -m pose_refine.evaluate_render_compare \
        --config pose_refine/configs/concat_loc_oh_v22d_stage2_lownoise.yaml \
        --gpu 0 --num_steps 100 --lr 0.001 --num_seeds 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

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
logger = logging.getLogger("render_compare")


def _pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray):
    """Rotation (degrees) and translation (metres) between two w2c matrices."""
    R_pred = pred_w2c[:3, :3]
    t_pred = pred_w2c[:3, 3]
    R_gt = gt_w2c[:3, :3]
    t_gt = gt_w2c[:3, 3]

    # Camera centres
    c_pred = -R_pred.T @ t_pred
    c_gt = -R_gt.T @ t_gt
    trans_err = float(np.linalg.norm(c_pred - c_gt))

    # Rotation angle
    dR = R_pred @ R_gt.T
    cos_angle = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.degrees(np.arccos(cos_angle)))
    return rot_err, trans_err


def render_features_differentiable(
    gaussians, dcff_renderer, feat_sharp, pose_w2c, K, render_h, render_w,
):
    """Render features at a given pose WITH gradients flowing through pose."""
    viewmat = pose_w2c.float()
    if viewmat.dim() == 2:
        viewmat = viewmat.unsqueeze(0)

    result = dcff_renderer(
        gaussians,
        viewmat=viewmat.squeeze(0),
        K=K,
        width=render_w,
        height=render_h,
        render_coarse=False,
    )
    fine_feat = result["fine_features"].float()
    depth = result["depth"]
    alpha = result.get("alpha")
    fine_feat = feat_sharp(
        fine_feat,
        depth=depth.float() if depth is not None else None,
        alpha=alpha.float() if alpha is not None else None,
    )
    return fine_feat, depth, alpha


def _compute_feature_loss(query_feat, ref_feat, alpha, loss_type, render_h, render_w, device):
    """Compute feature similarity loss between query and rendered features."""
    if alpha is not None:
        mask = (alpha > 0.5).float()
    else:
        mask = torch.ones(1, 1, render_h, render_w, device=device)

    if loss_type == "cosine":
        qf = F.normalize(query_feat, dim=1)
        rf = F.normalize(ref_feat, dim=1)
        sim = (qf * rf).sum(dim=1, keepdim=True) * mask
        return 1.0 - sim.sum() / mask.sum().clamp(min=1)
    else:  # l2
        diff = (query_feat - ref_feat) ** 2
        return (diff.sum(dim=1, keepdim=True) * mask).sum() / mask.sum().clamp(min=1)


def _render_and_loss(xi_np, init_pose_t, query_feat, gaussians, dcff_renderer,
                     feat_sharp, K, render_h, render_w, loss_type, device):
    """Render at pose defined by se3 delta and compute loss."""
    from pose_refine.utils.lie_algebra import se3_exp
    with torch.no_grad():
        d = torch.tensor(xi_np, device=device, dtype=torch.float32).unsqueeze(0)
        T = se3_exp(d)
        pose = torch.bmm(T, init_pose_t).squeeze(0)
        ref_feat, depth, alpha = render_features_differentiable(
            gaussians, dcff_renderer, feat_sharp, pose, K, render_h, render_w,
        )
        loss = _compute_feature_loss(
            query_feat, ref_feat, alpha, loss_type, render_h, render_w, device,
        )
    return loss.item()


def optimize_pose(
    query_feat: torch.Tensor,
    init_pose: torch.Tensor,
    gaussians,
    dcff_renderer,
    feat_sharp,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
    num_steps: int = 100,
    lr: float = 1e-3,
    loss_type: str = "l2",
):
    """Optimize pose via numerical-gradient descent on feature similarity.

    Uses finite-difference gradients since gsplat 2DGS doesn't backpropagate
    through viewmats. 6 DOF → 12 forward passes per step.

    Returns:
        optimized_pose: [4, 4] numpy array
        loss_history: list of loss values
    """
    from pose_refine.utils.lie_algebra import se3_exp

    device = query_feat.device
    init_pose_t = init_pose.float().to(device)
    if init_pose_t.dim() == 2:
        init_pose_t = init_pose_t.unsqueeze(0)

    xi = np.zeros(6, dtype=np.float64)
    eps = 1e-4  # finite difference epsilon

    loss_history = []
    best_loss = float("inf")
    best_xi = xi.copy()

    for step in range(num_steps):
        # Current loss
        curr_loss = _render_and_loss(
            xi, init_pose_t, query_feat, gaussians, dcff_renderer,
            feat_sharp, K, render_h, render_w, loss_type, device,
        )
        loss_history.append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_xi = xi.copy()

        # Numerical gradient (central differences)
        grad = np.zeros(6, dtype=np.float64)
        for d in range(6):
            xi_p = xi.copy(); xi_p[d] += eps
            xi_n = xi.copy(); xi_n[d] -= eps
            loss_p = _render_and_loss(
                xi_p, init_pose_t, query_feat, gaussians, dcff_renderer,
                feat_sharp, K, render_h, render_w, loss_type, device,
            )
            loss_n = _render_and_loss(
                xi_n, init_pose_t, query_feat, gaussians, dcff_renderer,
                feat_sharp, K, render_h, render_w, loss_type, device,
            )
            grad[d] = (loss_p - loss_n) / (2 * eps)

        # Adaptive step size with momentum-like decay
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        step_lr = lr / (1.0 + 0.01 * step)
        xi = xi - step_lr * grad

    # Return best pose
    with torch.no_grad():
        d = torch.tensor(best_xi, device=device, dtype=torch.float32).unsqueeze(0)
        T_best = se3_exp(d)
        final_pose = torch.bmm(T_best, init_pose_t)

    return final_pose.squeeze(0).cpu().numpy(), loss_history


def main():
    parser = argparse.ArgumentParser(description="Render-and-compare pose optimization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=100, help="Optimization steps per image")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "cosine"])
    parser.add_argument("--noise_deg", type=float, default=3.0, help="Rotation noise (degrees)")
    parser.add_argument("--noise_trans", type=float, default=0.1, help="Translation noise (metres)")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--max_test", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    from feature_field.utils.project_config import load_mainline_config
    from feature_field import build_dcff, intrinsics_to_K
    from pose_refine.utils.lie_algebra import se3_exp

    config = load_mainline_config(args.config)
    logger.info("Building DCFF rendering pipeline...")
    gaussians_dcff, dcff_renderer, feat_sharp = build_dcff(config, device)

    # Freeze all Gaussian params — we only optimize pose
    gaussians_dcff.set_geometry_trainable(False)
    if hasattr(gaussians_dcff, '_latent'):
        gaussians_dcff._latent.requires_grad_(False)
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

    # Load dataset
    from data.radio_loc_dataset import RadioLocDataset
    val_ds = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        split="train",  # "train" = test in Cambridge convention
        split_file=ds_cfg.get("test_split"),
        fine_hw=(render_h, render_w),
        coarse_hw=(render_h, render_w),
        noise_rot_deg=args.noise_deg,
        noise_trans_m=args.noise_trans,
        cache_in_memory=True,
    )

    # Intrinsics: scale from original image resolution to render resolution
    base_intr = val_ds.intrinsics  # dict with fx, fy, cx, cy at orig image res
    orig_h, orig_w = 1080, 1920  # Cambridge OldHospital standard resolution
    sx = render_w / orig_w
    sy = render_h / orig_h
    render_intr = {
        'fx': base_intr['fx'] * sx,
        'fy': base_intr['fy'] * sy,
        'cx': base_intr['cx'] * sx,
        'cy': base_intr['cy'] * sy,
    }
    K = intrinsics_to_K(render_intr, device)
    logger.info("Render: %d×%d  Steps: %d  LR: %.4f  Seeds: %d  Test: %d",
                render_w, render_h, args.num_steps, args.lr, args.num_seeds, len(val_ds))
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

            # Initial error
            i_rot, i_trans = _pose_errors(noisy_pose, gt_pose)
            init_rots.append(i_rot)
            init_trans.append(i_trans)

            # Optimize
            init_pose_t = torch.from_numpy(noisy_pose.astype(np.float32)).to(device)
            opt_pose, loss_hist = optimize_pose(
                query_fine, init_pose_t,
                gaussians_dcff, dcff_renderer, feat_sharp,
                K, render_h, render_w,
                num_steps=args.num_steps, lr=args.lr, loss_type=args.loss_type,
            )

            o_rot, o_trans = _pose_errors(opt_pose, gt_pose)
            opt_rots.append(o_rot)
            opt_trans.append(o_trans)

            if (i + 1) % 10 == 0 or i + 1 == n_test:
                ir = np.array(init_rots)
                it_arr = np.array(init_trans) * 1000
                opr = np.array(opt_rots)
                opt_arr = np.array(opt_trans) * 1000
                logger.info(
                    "  [%3d/%d] seed=%d  init=%.2f°/%.0fmm  opt=%.2f°/%.0fmm  Δ=%.0fmm",
                    i + 1, n_test, seed,
                    np.median(ir), np.median(it_arr),
                    np.median(opr), np.median(opt_arr),
                    np.median(opt_arr) - np.median(it_arr),
                )

        elapsed = time.time() - t_start
        ir_arr = np.array(init_rots)
        it_arr = np.array(init_trans) * 1000
        or_arr = np.array(opt_rots)
        ot_arr = np.array(opt_trans) * 1000

        logger.info(
            "[seed %d] %d steps  init=%.2f°/%.0fmm  opt=%.2f°/%.0fmm  (%.1fs)",
            seed, args.num_steps,
            np.median(ir_arr), np.median(it_arr),
            np.median(or_arr), np.median(ot_arr),
            elapsed,
        )

        all_seed_results.append({
            "seed": seed,
            "init_rot_med": float(np.median(ir_arr)),
            "init_trans_med_mm": float(np.median(it_arr)),
            "opt_rot_med": float(np.median(or_arr)),
            "opt_trans_med_mm": float(np.median(ot_arr)),
            "time_s": round(elapsed, 1),
        })

    # Summary
    print("\n" + "=" * 70)
    print("RENDER-AND-COMPARE OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Noise: {args.noise_deg}° / {args.noise_trans}m")
    print(f"Steps: {args.num_steps}  LR: {args.lr}  Loss: {args.loss_type}")
    print(f"Seeds: {args.num_seeds}")
    print()

    for r in all_seed_results:
        print(f"  seed {r['seed']}: init={r['init_rot_med']:.2f}°/{r['init_trans_med_mm']:.0f}mm  "
              f"opt={r['opt_rot_med']:.2f}°/{r['opt_trans_med_mm']:.0f}mm  "
              f"({r['time_s']:.0f}s)")

    if len(all_seed_results) > 1:
        avg_rot = np.mean([r["opt_rot_med"] for r in all_seed_results])
        avg_trans = np.mean([r["opt_trans_med_mm"] for r in all_seed_results])
        print(f"\n  Average: opt={avg_rot:.2f}°/{avg_trans:.0f}mm")

    print(f"\n  SOTA (GSFFs): 0.32° / 183mm")
    print("=" * 70)

    # Save results
    out_dir = os.path.join(_PROJ_ROOT, "output", "render_compare_eval")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(
        out_dir, f"results_n{args.num_steps}_lr{args.lr}_{args.loss_type}_"
                 f"noise{args.noise_deg}deg_{args.noise_trans}m.json"
    )
    with open(result_path, "w") as f:
        json.dump({
            "config": vars(args),
            "seeds": all_seed_results,
        }, f, indent=2)
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
