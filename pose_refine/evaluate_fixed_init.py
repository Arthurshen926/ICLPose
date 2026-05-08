#!/usr/bin/env python3
"""Fixed-init evaluation: measure refinement per noise bucket.

Evaluates Camera Pose Refinement: given a GT pose + controlled synthetic noise,
how much does the refiner improve the pose? Reports per-bucket and aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.radio_loc_dataset import RadioLocDataset, add_pose_noise
from pose_refine.evaluate_impl import (
    build_dcff,
    camera_centers_from_w2c,
    intrinsics_to_K,
    load_model,
    render_batch,
)
from pose_refine.utils.lie_algebra import se3_exp


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-init CPR evaluation")
    parser.add_argument("--config", required=True, help="Mainline YAML config")
    parser.add_argument("--checkpoint", required=True, help="Pose refiner checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--buckets", type=str,
                        default="0.1,2 0.25,5 0.5,10 1.0,20 2.0,30",
                        help="Space-separated trans_m,rot_deg pairs")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of noise seeds per bucket per image")
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--gru-iters", type=int, default=4)
    parser.add_argument("--render-h", type=int, default=68)
    parser.add_argument("--render-w", type=int, default=120)
    parser.add_argument("--use-coarse", action="store_true", default=True)
    return parser.parse_args()


def parse_buckets(s: str) -> List[Tuple[float, float]]:
    buckets = []
    for pair in s.strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            buckets.append((float(parts[0]), float(parts[1])))
    return buckets


def compute_pose_error(pose_pred: torch.Tensor, pose_gt: torch.Tensor):
    """Return (rot_err_deg, trans_err_mm) between two w2c poses."""
    R_pred = pose_pred[:3, :3]
    R_gt = pose_gt[:3, :3]
    t_pred = pose_pred[:3, 3]
    t_gt = pose_gt[:3, 3]

    R_rel = R_pred.T @ R_gt
    trace = torch.clamp(R_rel.trace(), -1.0, 3.0)
    rot_err = torch.acos((trace - 1.0) / 2.0).item() * 180.0 / math.pi

    C_pred = -(R_pred.T @ t_pred)
    C_gt = -(R_gt.T @ t_gt)
    trans_err = (C_pred - C_gt).norm().item() * 1000.0

    return rot_err, trans_err


def evaluate_on_bucket(model, gaussians, dcff_renderer, feat_sharp,
                        dataset, bucket_trans_m, bucket_rot_deg,
                        n_seeds, device, args, K, render_intr):
    """Evaluate refinement on a single noise bucket."""
    all_init_rot, all_init_trans = [], []
    all_final_rot, all_final_trans = [], []

    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = args.gru_iters
    update_scale = float(getattr(model, 'pose_update_scale', 1.0))

    for idx in tqdm(range(len(dataset)),
                     desc=f"bucket {bucket_trans_m:.2f}m/{bucket_rot_deg:.0f}deg"):
        sample = dataset[idx]
        query_fine = sample["query_fine"].unsqueeze(0).to(device)
        query_coarse = sample.get("query_coarse")
        if query_coarse is not None and args.use_coarse:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_coarse = None
        pose_gt = sample["pose_gt"].to(device)

        for seed in range(n_seeds):
            pose_init_np = add_pose_noise(
                pose_gt.cpu().numpy(), bucket_rot_deg, bucket_trans_m
            )
            pose_init = torch.from_numpy(pose_init_np).float().to(device)

            init_rot, init_trans = compute_pose_error(pose_init, pose_gt)
            all_init_rot.append(init_rot)
            all_init_trans.append(init_trans)

            pose_cur = pose_init.unsqueeze(0)
            for outer_i in range(args.outer_iters):
                ref_fine, depth = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, args.render_h, args.render_w,
                )

                with autocast(enabled=True):
                    pred = model(
                        query_fine, ref_fine, depth,
                        intrinsics=render_intr,
                        query_coarse=query_coarse,
                    )

                if 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float() * update_scale)
                    pose_cur = torch.bmm(T_delta, pose_cur.float())

            final_rot, final_trans = compute_pose_error(
                pose_cur.squeeze(0), pose_gt
            )
            all_final_rot.append(final_rot)
            all_final_trans.append(final_trans)

    model.gru_iters = original_gru

    return {
        "init_rot_deg_mean": float(np.mean(all_init_rot)),
        "init_trans_mm_mean": float(np.mean(all_init_trans)),
        "init_trans_mm_median": float(np.median(all_init_trans)),
        "final_rot_deg_mean": float(np.mean(all_final_rot)),
        "final_trans_mm_mean": float(np.mean(all_final_trans)),
        "final_trans_mm_median": float(np.median(all_final_trans)),
        "trans_gain_mm": float(np.mean(all_init_trans) - np.mean(all_final_trans)),
        "rot_gain_deg": float(np.mean(all_init_rot) - np.mean(all_final_rot)),
        "n_samples": len(all_init_trans),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    model.eval()

    # Set intrinsics on model (needed for _scale_intrinsics)
    from data.radio_loc_dataset import read_colmap_cameras, camera_params_to_intrinsics
    colmap_dir = config["dataset"]["colmap_dir"]
    colmap_cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
    first_cam = next(iter(colmap_cameras.values()))
    model.BASE_INTRINSICS = camera_params_to_intrinsics(first_cam)
    model.IMG_HW = (int(first_cam.height), int(first_cam.width))

    render_intr = model._scale_intrinsics(args.render_h, args.render_w)
    K = intrinsics_to_K(render_intr, device)

    buckets = parse_buckets(args.buckets)

    dataset = RadioLocDataset(
        feature_dir=config["dataset"]["feature_dir"],
        colmap_dir=config["dataset"]["colmap_dir"],
        split=args.split,
        split_file=config["dataset"].get(f"{args.split}_split"),
        noise_rot_deg=0.0,
        noise_trans_m=0.0,
    )
    if args.limit is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    results = {}
    for trans_m, rot_deg in buckets:
        bucket_key = f"{trans_m:.2f}m_{rot_deg:.0f}deg"
        results[bucket_key] = evaluate_on_bucket(
            model, gaussians, dcff_renderer, feat_sharp,
            dataset, trans_m, rot_deg, args.n_seeds, device, args, K, render_intr
        )

    print("\n=== Fixed-Init CPR Results ===")
    print(f"{'Bucket':<20s} {'Init(mm)':>10s} {'Final(mm)':>10s} {'Gain(mm)':>10s} {'Gain%':>8s}  {'Init(deg)':>10s} {'Final(deg)':>10s}")
    print("-" * 78)
    for bucket_key, r in results.items():
        gain_pct = r["trans_gain_mm"] / max(r["init_trans_mm_mean"], 1e-6) * 100
        print(f"{bucket_key:<20s} {r['init_trans_mm_median']:>10.1f} "
              f"{r['final_trans_mm_median']:>10.1f} {r['trans_gain_mm']:>10.1f} "
              f"{gain_pct:>7.1f}%  {r['init_rot_deg_mean']:>10.2f} {r['final_rot_deg_mean']:>10.2f}")

    output_path = args.output_json or os.path.join(
        os.path.dirname(args.checkpoint), "fixed_init_results.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
