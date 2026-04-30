#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import RadioLocDataset, collate_fn, read_colmap_cameras
from feature_field import build_dcff_runtime, intrinsics_to_K, render_feature_bundle_batch
from pose_refine.models.concat_pose_net import local_correlation
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
    trans = torch.norm(
        camera_centers_from_w2c(pose_pred) - camera_centers_from_w2c(pose_gt),
        dim=1,
    ) * 1000.0
    return rot * 180.0 / math.pi, trans


def scale_intrinsics(base_intr: dict, orig_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict:
    orig_h, orig_w = orig_hw
    h, w = target_hw
    return {
        "fx": float(base_intr["fx"] * w / orig_w),
        "fy": float(base_intr["fy"] * h / orig_h),
        "cx": float(base_intr["cx"] * w / orig_w),
        "cy": float(base_intr["cy"] * h / orig_h),
    }


def soft_corr_flow(
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    radius: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r = F.normalize(rendered_feat.float(), dim=1)
    q = F.normalize(query_feat.float(), dim=1)
    corr = local_correlation(r, q, radius=radius)
    B, _, H, W = corr.shape
    win = 2 * radius + 1

    weights = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=corr.device, dtype=torch.float32),
        torch.arange(-radius, radius + 1, device=corr.device, dtype=torch.float32),
        indexing="ij",
    )
    offsets_x = offsets_x.reshape(1, win * win, 1, 1)
    offsets_y = offsets_y.reshape(1, win * win, 1, 1)

    flow_u = (weights * offsets_x).sum(dim=1, keepdim=True)
    flow_v = (weights * offsets_y).sum(dim=1, keepdim=True)
    flow = torch.cat([flow_u, flow_v], dim=1)
    confidence = weights.max(dim=1, keepdim=True).values
    return flow, confidence


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot local-correlation flow + WLS evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--noise_deg", type=float, default=1.0)
    parser.add_argument("--noise_m", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--outer_iters", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.05)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    cameras = read_colmap_cameras(str(Path(ds_cfg["colmap_dir"]) / "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    render_intr = scale_intrinsics(dataset.intrinsics, (int(first_cam.height), int(first_cam.width)), (render_h, render_w))
    K = intrinsics_to_K(render_intr, device)

    print(
        f"Render: {render_w}x{render_h}  noise={args.noise_deg}deg/{args.noise_m}m  "
        f"radius={args.radius} temp={args.temperature}"
    )

    for max_oi in args.outer_iters:
        init_rot_all, init_trans_all = [], []
        final_rot_all, final_trans_all = [], []
        flow_mag_all, conf_all = [], []

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"corr-wls oi={max_oi}", leave=False)):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            query_fine = batch["query_fine"].to(device).float()
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
                rendered = bundle["fine_features"].float()
                depth = bundle["depth"]
                if depth.ndim == 4:
                    depth = depth.squeeze(1)

                flow, confidence = soft_corr_flow(rendered, query_fine, args.radius, args.temperature)
                flow_mag_all.append(float(flow.norm(dim=1).mean().item()))
                conf_all.append(float(confidence.mean().item()))

                Ju, Jv, valid = compute_image_jacobian(depth.float(), render_intr)
                delta_xi = diff_pose_solve(
                    flow.float(),
                    confidence.float(),
                    Ju,
                    Jv,
                    valid,
                    damping=1e-3,
                    irls_iters=args.irls_iters,
                )
                pose_cur = torch.bmm(se3_exp(delta_xi.float()), pose_cur.float())

            final_rot, final_trans = pose_errors(pose_cur, pose_gt)
            final_rot_all.extend(final_rot.cpu().tolist())
            final_trans_all.extend(final_trans.cpu().tolist())

        print(
            f"oi={max_oi}: init_med={np.median(init_rot_all):.3f}deg/"
            f"{np.median(init_trans_all):.1f}mm  final_med={np.median(final_rot_all):.3f}deg/"
            f"{np.median(final_trans_all):.1f}mm  flow_mag={np.mean(flow_mag_all):.2f}px  "
            f"conf={np.mean(conf_all):.3f}"
        )


if __name__ == "__main__":
    main()
