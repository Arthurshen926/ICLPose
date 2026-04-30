#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import RadioLocDataset, collate_fn, read_colmap_cameras
from feature_field import build_dcff_runtime, intrinsics_to_K, render_feature_bundle_batch
from pose_refine.models.concat_pose_net import ConcatPoseNet
from pose_refine.utils.geometry_solver import compute_image_jacobian, diff_pose_solve
from pose_refine.utils.lie_algebra import se3_exp


def camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def pose_errors(pose_pred: torch.Tensor, pose_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    R_rel = torch.bmm(pose_pred[:, :3, :3].transpose(1, 2), pose_gt[:, :3, :3])
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    rot = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    c_pred = camera_centers_from_w2c(pose_pred)
    c_gt = camera_centers_from_w2c(pose_gt)
    trans = torch.norm(c_pred - c_gt, dim=1) * 1000.0
    return rot * 180.0 / math.pi, trans


def scale_intrinsics(base_intr: dict, orig_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict:
    orig_h, orig_w = orig_hw
    h, w = target_hw
    sx = w / orig_w
    sy = h / orig_h
    return {
        "fx": float(base_intr["fx"] * sx),
        "fy": float(base_intr["fy"] * sy),
        "cx": float(base_intr["cx"] * sx),
        "cy": float(base_intr["cy"] * sy),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-flow WLS diagnostic for current DCFF localization stack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--noise_deg", type=float, default=1.0)
    parser.add_argument("--noise_m", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--outer_iters", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--irls_iters", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runtime = build_dcff_runtime(config, device, printer=print)
    render_h = int(runtime.render_height)
    render_w = int(runtime.render_width)

    ds_cfg = config["dataset"]
    dataset = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        source_dir=ds_cfg.get("source_dir"),
        split="test",
        split_file=ds_cfg.get("test_split"),
        noise_rot_deg=args.noise_deg,
        noise_trans_m=args.noise_m,
        coarse_hw=tuple(ds_cfg.get("coarse_hw", [render_h, render_w])),
        fine_hw=tuple(ds_cfg.get("fine_hw", [render_h, render_w])),
        cache_in_memory=False,
        normalize_features=ds_cfg.get("normalize_features", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    cameras = read_colmap_cameras(str(Path(ds_cfg["colmap_dir"]) / "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    orig_hw = (int(first_cam.height), int(first_cam.width))
    render_intr = scale_intrinsics(dataset.intrinsics, orig_hw, (render_h, render_w))
    K = intrinsics_to_K(render_intr, device)

    print(f"Render: {render_w}x{render_h}  noise={args.noise_deg}deg/{args.noise_m}m")
    print(f"outer_iters={args.outer_iters}  irls_iters={args.irls_iters}")

    for use_gt_valid in (False, True):
        mask_name = "gt_valid_mask" if use_gt_valid else "depth_only_mask"
        print(f"\n[{mask_name}]")
        for max_oi in args.outer_iters:
            init_rot_all, init_trans_all = [], []
            final_rot_all, final_trans_all = [], []
            valid_ratio_all = []

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"{mask_name} oi={max_oi}", leave=False)):
                if args.max_batches > 0 and batch_idx >= args.max_batches:
                    break

                pose_gt = batch["pose_gt"].to(device).float()
                pose_cur = batch["pose_init"].to(device).float()

                init_rot, init_trans = pose_errors(pose_cur, pose_gt)
                init_rot_all.extend(init_rot.cpu().tolist())
                init_trans_all.extend(init_trans.cpu().tolist())

                for _ in range(max_oi):
                    bundle = render_feature_bundle_batch(
                        runtime.gaussians,
                        runtime.renderer,
                        runtime.refiner,
                        pose_cur,
                        K,
                        render_h,
                        render_w,
                        render_coarse=False,
                    )
                    depth = bundle["depth"]
                    if depth.ndim == 4:
                        depth = depth.squeeze(1)

                    gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
                        pose_cur,
                        pose_gt,
                        depth,
                        (render_h, render_w),
                        render_intr,
                    )
                    Ju, Jv, depth_valid = compute_image_jacobian(depth.float(), render_intr)
                    if use_gt_valid:
                        solve_valid = depth_valid & (gt_valid.squeeze(1).reshape(depth.shape[0], -1) > 0.5)
                        confidence = gt_valid.float()
                    else:
                        solve_valid = depth_valid
                        confidence = torch.ones(
                            depth.shape[0], 1, render_h, render_w,
                            device=device, dtype=torch.float32,
                        )

                    valid_ratio_all.append(float(solve_valid.float().mean().item()))
                    delta_xi = diff_pose_solve(
                        gt_flow.float(),
                        confidence,
                        Ju,
                        Jv,
                        solve_valid,
                        damping=1e-3,
                        irls_iters=args.irls_iters,
                    )
                    pose_cur = torch.bmm(se3_exp(delta_xi.float()), pose_cur.float())

                final_rot, final_trans = pose_errors(pose_cur, pose_gt)
                final_rot_all.extend(final_rot.cpu().tolist())
                final_trans_all.extend(final_trans.cpu().tolist())

            print(
                f"  oi={max_oi}: init_med={np.median(init_rot_all):.3f}deg/"
                f"{np.median(init_trans_all):.1f}mm  final_med={np.median(final_rot_all):.3f}deg/"
                f"{np.median(final_trans_all):.1f}mm  valid={np.mean(valid_ratio_all) * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
