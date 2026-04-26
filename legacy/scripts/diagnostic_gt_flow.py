#!/usr/bin/env python3
"""
Diagnostic: Does diff_pose_solve recover pose from PERFECT (GT) flow?
=====================================================================

Tests the geometry solver in isolation to determine whether pose recovery
failure is due to the solver or the flow prediction network.

Pipeline per noise level:
  1. Load one sample from RadioLocDataset (GT pose + intrinsics)
  2. Apply known noise to get an initial pose
  3. Render depth at the initial pose via GaussianFeatureModel + FeatureRenderer
  4. Compute GT flow from initial→GT using lie_algebra.compute_gt_flow
  5. Feed GT flow + depth → compute_image_jacobian → diff_pose_solve → Δξ
  6. Apply: T_updated = se3_exp(Δξ) @ T_init
  7. Measure rotation & translation error vs GT

If the solver recovers the pose (errors ≈ 0), the localization problem
is purely in flow prediction quality.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/diagnostic_gt_flow.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import (
    RadioLocDataset,
    add_pose_noise,
    camera_params_to_intrinsics,
)
from modules.geometry_solver import compute_image_jacobian, diff_pose_solve
from modules.lie_algebra import se3_exp, compute_gt_flow
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def pose_error(pred_w2c: torch.Tensor, gt_w2c: torch.Tensor):
    """Compute rotation (deg) and translation (m) error between w2c poses."""
    pred_c2w = torch.inverse(pred_w2c)
    gt_c2w = torch.inverse(gt_w2c)

    # Translation error (metres)
    trans_err = (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item()

    # Rotation error (degrees)
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = R_rel.diagonal().sum()
    cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    rot_err = torch.acos(cos_a).item() * 180.0 / math.pi

    return rot_err, trans_err


def load_renderer(config_path: str, device: torch.device):
    """Load GaussianFeatureModel for depth rendering from DCFF config.

    Only geometry is needed (no DCFF decoder weights) — just the PLY file.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ply_path = cfg['dcff']['ply_path']
    gs_model = GaussianFeatureModel(feature_dim=64)
    gs_model.load_ply(ply_path)
    gs_model = gs_model.to(device)
    gs_model.eval()

    return gs_model, cfg


def render_depth_at_pose(
    gs_model: GaussianFeatureModel,
    pose_w2c: torch.Tensor,
    intrinsics: dict,
    hw: tuple,
):
    """Render depth map at a given w2c pose. Returns (1, 1, H, W)."""
    H, W = hw
    orig_H, orig_W = 1080, 1920
    sx = W / orig_W
    sy = H / orig_H

    depth_map = FeatureRenderer.render_depth(
        gs_model,
        pose_w2c,  # (4, 4)
        fx=intrinsics['fx'] * sx,
        fy=intrinsics['fy'] * sy,
        cx=intrinsics['cx'] * sx,
        cy=intrinsics['cy'] * sy,
        img_height=H,
        img_width=W,
    )  # (H, W)
    return depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config_path = 'configs/radio_loc_oh_v3.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load renderer (GaussianFeatureModel for depth) ──
    print("\n[1] Loading Gaussian model for depth rendering...")
    gs_model, _ = load_renderer(config_path, device)

    # ── Load one dataset sample ──
    print("[2] Loading dataset (train split, 1 sample)...")
    ds_cfg = cfg['dataset']
    dataset = RadioLocDataset(
        feature_dir=ds_cfg['feature_dir'],
        colmap_dir=ds_cfg['colmap_dir'],
        split='train',
        split_file=ds_cfg['train_split'],
        noise_rot_deg=0.0,   # no noise — we add our own
        noise_trans_m=0.0,
        coarse_hw=cfg['model']['coarse_hw'],
        fine_hw=cfg['model']['fine_hw'],
        cache_in_memory=False,
    )

    sample = dataset[0]
    pose_gt_w2c = sample['pose_gt'].to(device)   # (4, 4)
    raw_intrinsics = dataset.intrinsics           # {fx, fy, cx, cy} at original resolution

    fine_hw = tuple(cfg['model']['fine_hw'])       # (68, 120)
    H, W = fine_hw
    orig_H, orig_W = 1080, 1920
    sx = W / orig_W
    sy = H / orig_H

    scaled_intrinsics = {
        'fx': raw_intrinsics['fx'] * sx,
        'fy': raw_intrinsics['fy'] * sy,
        'cx': raw_intrinsics['cx'] * sx,
        'cy': raw_intrinsics['cy'] * sy,
    }

    print(f"   GT pose loaded. Image resolution: {H}×{W}")
    print(f"   Scaled intrinsics: fx={scaled_intrinsics['fx']:.2f}, "
          f"fy={scaled_intrinsics['fy']:.2f}, "
          f"cx={scaled_intrinsics['cx']:.2f}, "
          f"cy={scaled_intrinsics['cy']:.2f}")

    # ── Noise levels to test ──
    noise_levels = [
        (1.0, 0.02),
        (5.0, 0.20),
        (10.0, 0.50),
    ]

    # Build K matrix for diff_pose_solve
    K = torch.eye(3, device=device).unsqueeze(0)  # (1, 3, 3)
    K[0, 0, 0] = scaled_intrinsics['fx']
    K[0, 1, 1] = scaled_intrinsics['fy']
    K[0, 0, 2] = scaled_intrinsics['cx']
    K[0, 1, 2] = scaled_intrinsics['cy']

    print("\n" + "=" * 80)
    print("DIAGNOSTIC: GT Flow → diff_pose_solve → Pose Recovery")
    print("=" * 80)

    for rot_deg, trans_m in noise_levels:
        print(f"\n{'─' * 70}")
        print(f"  Noise level: {rot_deg}° / {trans_m}m")
        print(f"{'─' * 70}")

        # Apply noise to GT pose (deterministic seed for reproducibility)
        np.random.seed(42)
        pose_init_np = add_pose_noise(
            pose_gt_w2c.cpu().numpy(), rot_deg, trans_m
        )
        pose_init_w2c = torch.from_numpy(pose_init_np).float().to(device)

        # Measure initial error
        init_rot_err, init_trans_err = pose_error(pose_init_w2c, pose_gt_w2c)
        print(f"  Initial error:  rot={init_rot_err:.4f}°  trans={init_trans_err:.4f}m")

        # ── Render depth at initial pose ──
        depth = render_depth_at_pose(
            gs_model, pose_init_w2c, raw_intrinsics, fine_hw
        )  # (1, 1, H, W)

        valid_pct = (depth > 0.05).float().mean().item() * 100
        print(f"  Depth at init pose: min={depth.min():.3f}, "
              f"max={depth.max():.3f}, valid={valid_pct:.1f}%")

        # ── Compute GT flow: from init pose to GT pose ──
        # compute_gt_flow: takes depth at pose_gt, computes flow from GT → current
        # But we need flow at the INIT pose that maps pixels to where they'd be at GT.
        #
        # Recompute: We have depth at init pose.
        # For each pixel (u,v) in init view at depth z:
        #   unproject to camera coords → world coords (via c2w_init) → reproject via w2c_gt → (u', v')
        #   flow = (u' - u, v' - v)
        #
        # compute_gt_flow in lie_algebra.py does exactly this:
        #   depth_gt = depth at pose_gt frame, flow from GT frame → current frame
        # We need the reverse: depth at init frame, flow from init frame → GT frame.
        #
        # The compute_gt_flow(depth_gt, pose_gt, pose_current, intrinsics) computes:
        #   unproject pixels from pose_gt frame → world → project to pose_current frame
        # So for "flow from init → GT", call:
        #   compute_gt_flow(depth_init, pose_init, pose_gt, intrinsics)
        #   This unprojects from init frame → world → projects to GT frame → flow
        gt_flow_result = compute_gt_flow(
            depth_gt=depth.squeeze(0).squeeze(0),     # (H, W)
            pose_gt=pose_init_w2c,                     # (4, 4) - "source" frame
            pose_current=pose_gt_w2c,                  # (4, 4) - "target" frame
            intrinsics=scaled_intrinsics,
        )
        gt_flow = gt_flow_result['flow'].unsqueeze(0)        # (1, 2, H, W)
        valid_mask = gt_flow_result['valid_mask'].unsqueeze(0)  # (1, 1, H, W)

        flow_mag = gt_flow.norm(dim=1).mean().item()
        print(f"  GT flow: mean magnitude={flow_mag:.3f}px, "
              f"valid={valid_mask.mean().item()*100:.1f}%")

        # ── Compute Image Jacobian at init pose depth ──
        depth_for_jacobian = depth.squeeze(1)  # (1, H, W)
        Ju, Jv, valid = compute_image_jacobian(depth_for_jacobian, scaled_intrinsics)

        # ── Run diff_pose_solve (single step, no IRLS) ──
        confidence = valid_mask  # (1, 1, H, W) — uniform confidence on valid pixels
        delta_xi = diff_pose_solve(
            flow=gt_flow,
            confidence=confidence,
            Ju=Ju,
            Jv=Jv,
            valid=valid,
            damping=1e-4,
            irls_iters=0,
        )

        print(f"  Solved Δξ: trans=[{delta_xi[0,0]:.5f}, {delta_xi[0,1]:.5f}, {delta_xi[0,2]:.5f}], "
              f"rot=[{delta_xi[0,3]:.5f}, {delta_xi[0,4]:.5f}, {delta_xi[0,5]:.5f}]")

        # ── Apply pose update: T_new = exp(Δξ) @ T_init ──
        delta_T = se3_exp(delta_xi)             # (1, 4, 4)
        pose_updated_w2c = (delta_T @ pose_init_w2c.unsqueeze(0)).squeeze(0)

        # ── Measure final error ──
        final_rot_err, final_trans_err = pose_error(pose_updated_w2c, pose_gt_w2c)
        print(f"  Updated error:  rot={final_rot_err:.4f}°  trans={final_trans_err:.4f}m")

        rot_reduction = (1 - final_rot_err / max(init_rot_err, 1e-8)) * 100
        trans_reduction = (1 - final_trans_err / max(init_trans_err, 1e-8)) * 100
        print(f"  Reduction:      rot={rot_reduction:.1f}%  trans={trans_reduction:.1f}%")

        # ── Also test with IRLS ──
        delta_xi_irls = diff_pose_solve(
            flow=gt_flow,
            confidence=confidence,
            Ju=Ju,
            Jv=Jv,
            valid=valid,
            damping=1e-4,
            irls_iters=3,
            robust_kernel='huber',
        )
        delta_T_irls = se3_exp(delta_xi_irls)
        pose_irls = (delta_T_irls @ pose_init_w2c.unsqueeze(0)).squeeze(0)
        irls_rot, irls_trans = pose_error(pose_irls, pose_gt_w2c)
        print(f"  IRLS (3 iter):  rot={irls_rot:.4f}°  trans={irls_trans:.4f}m")

        # ── Multi-step iterative refinement ──
        pose_iter = pose_init_w2c.clone()
        print(f"  Multi-step refinement (5 iterations):")
        for step in range(5):
            depth_iter = render_depth_at_pose(
                gs_model, pose_iter, raw_intrinsics, fine_hw
            )
            flow_iter_result = compute_gt_flow(
                depth_gt=depth_iter.squeeze(0).squeeze(0),
                pose_gt=pose_iter,
                pose_current=pose_gt_w2c,
                intrinsics=scaled_intrinsics,
            )
            flow_iter = flow_iter_result['flow'].unsqueeze(0)
            mask_iter = flow_iter_result['valid_mask'].unsqueeze(0)

            Ju_i, Jv_i, valid_i = compute_image_jacobian(
                depth_iter.squeeze(1), scaled_intrinsics
            )
            xi_i = diff_pose_solve(
                flow=flow_iter,
                confidence=mask_iter,
                Ju=Ju_i,
                Jv=Jv_i,
                valid=valid_i,
                damping=1e-4,
                irls_iters=0,
            )
            dT_i = se3_exp(xi_i)
            pose_iter = (dT_i @ pose_iter.unsqueeze(0)).squeeze(0)
            r_err, t_err = pose_error(pose_iter, pose_gt_w2c)
            flow_mag_i = flow_iter.norm(dim=1).mean().item()
            print(f"    step {step+1}: rot={r_err:.4f}°  trans={t_err:.4f}m  "
                  f"flow_mag={flow_mag_i:.3f}px")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  • If single-step errors are small → solver works; problem is flow quality.")
    print("  • If single-step errors are large → solver has a bug or is ill-conditioned.")
    print("  • Multi-step should converge to near-zero regardless of noise level.")
    print("=" * 80)


if __name__ == '__main__':
    run_diagnostic()
