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
from pose_refine.runtime import build_concat_pose_model
from pose_refine.train_impl import (
    feature_metric_pose_update,
    masked_feature_cosine_distance_per_sample,
    perturb_w2c_camera_center,
)


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


def render_bundle(runtime, poses, K, render_h, render_w):
    return render_feature_bundle_batch(
        runtime.gaussians,
        runtime.renderer,
        runtime.refiner,
        poses,
        K,
        render_h,
        render_w,
        render_coarse=False,
    )


def project_feature_pair(model, config, query_feat, rendered_feat):
    query = query_feat.float()
    rendered = rendered_feat.float()
    if query.shape[-2:] != rendered.shape[-2:]:
        query = F.interpolate(query, rendered.shape[-2:], mode="bilinear", align_corners=False)
    mode = config.get("model", {}).get("proj_mode", "separate")
    if mode == "shared":
        query = model.proj_shared(query)
        rendered = model.proj_shared(rendered)
    else:
        query = model.proj_query(query)
        rendered = model.proj_render(rendered)
    return F.normalize(query.float(), dim=1), F.normalize(rendered.float(), dim=1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="DCFF centimetre translation sensitivity diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--dist_cm", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--frame", default="camera", choices=["camera", "world"])
    parser.add_argument("--noise_deg", type=float, default=1.0)
    parser.add_argument("--noise_m", type=float, default=0.05)
    parser.add_argument("--fm_damping", type=float, default=1e-3)
    parser.add_argument("--fm_scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--pose_checkpoint", default=None)
    parser.add_argument("--use_projection", action="store_true")
    parser.add_argument("--no_feature_metric", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runtime = build_dcff_runtime(config, device, printer=print)
    render_h = int(runtime.render_height)
    render_w = int(runtime.render_width)
    ds_cfg = config["dataset"]
    split_file = ds_cfg.get("test_split") if args.split == "test" else ds_cfg.get("train_split")
    dataset = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        source_dir=ds_cfg.get("source_dir"),
        split=args.split,
        split_file=split_file,
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
    render_intr = scale_intrinsics(
        dataset.intrinsics,
        (int(first_cam.height), int(first_cam.width)),
        (render_h, render_w),
    )
    K = intrinsics_to_K(render_intr, device)
    proj_model = None
    if args.use_projection or args.pose_checkpoint:
        proj_model = build_concat_pose_model(config.get("model", {}), device)
        proj_model.BASE_INTRINSICS = dataset.intrinsics
        proj_model.IMG_HW = (int(first_cam.height), int(first_cam.width))
        if args.pose_checkpoint:
            ckpt = torch.load(args.pose_checkpoint, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            proj_model.load_state_dict(state, strict=False)
        proj_model.eval()

    pos_all = []
    rank_stats = {float(cm): {"neg": [], "acc": [], "gap": []} for cm in args.dist_cm}
    init_rot_all, init_trans_all = [], []
    fm_stats = {float(scale): {"rot": [], "trans": [], "delta": []} for scale in args.fm_scales}

    print(
        f"Render: {render_w}x{render_h}  split={args.split}  "
        f"noise={args.noise_deg}deg/{args.noise_m}m  frame={args.frame}"
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc="dcff-cm", leave=False)):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        query_fine = batch["query_fine"].to(device).float()
        pose_gt = batch["pose_gt"].to(device).float()
        pose_init = batch["pose_init"].to(device).float()

        gt_bundle = render_bundle(runtime, pose_gt, K, render_h, render_w)
        gt_mask = (gt_bundle["depth"].float() > 0.05).unsqueeze(1)
        pos_dist = masked_feature_cosine_distance_per_sample(
            query_fine,
            gt_bundle["fine_features"].float(),
            gt_mask,
        )
        pos_all.extend(pos_dist.cpu().tolist())

        for cm in args.dist_cm:
            dist_m = float(cm) / 100.0
            neg_dists = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    offsets = torch.zeros(pose_gt.shape[0], 3, device=device)
                    offsets[:, axis] = sign * dist_m
                    pose_neg = perturb_w2c_camera_center(pose_gt, offsets, frame=args.frame)
                    neg_bundle = render_bundle(runtime, pose_neg, K, render_h, render_w)
                    neg_mask = gt_mask * (neg_bundle["depth"].float() > 0.05).unsqueeze(1)
                    neg_dist = masked_feature_cosine_distance_per_sample(
                        query_fine,
                        neg_bundle["fine_features"].float(),
                        neg_mask,
                    )
                    neg_dists.append(neg_dist)
            neg_stack = torch.stack(neg_dists, dim=0)
            hard_neg = neg_stack.min(dim=0).values
            gap = hard_neg - pos_dist
            rank_stats[float(cm)]["neg"].extend(hard_neg.cpu().tolist())
            rank_stats[float(cm)]["gap"].extend(gap.cpu().tolist())
            rank_stats[float(cm)]["acc"].extend((gap > 0.0).float().cpu().tolist())

        if not args.no_feature_metric:
            init_rot, init_trans = pose_errors(pose_init, pose_gt)
            init_rot_all.extend(init_rot.cpu().tolist())
            init_trans_all.extend(init_trans.cpu().tolist())
            init_bundle = render_bundle(runtime, pose_init, K, render_h, render_w)
            init_mask = (init_bundle["depth"].float() > 0.05).unsqueeze(1)
            fm_query = query_fine
            fm_rendered = init_bundle["fine_features"].float()
            fm_normalize = True
            if proj_model is not None:
                fm_query, fm_rendered = project_feature_pair(
                    proj_model,
                    config,
                    query_fine,
                    fm_rendered,
                )
                fm_normalize = False
            for scale in args.fm_scales:
                _delta, fm_pose, _residual = feature_metric_pose_update(
                    fm_query,
                    fm_rendered,
                    init_bundle["depth"].float(),
                    pose_init,
                    render_intr,
                    damping=args.fm_damping,
                    valid_mask=init_mask,
                    normalize_features=fm_normalize,
                    update_scale=float(scale),
                )
                fm_rot, fm_trans = pose_errors(fm_pose, pose_gt)
                stat = fm_stats[float(scale)]
                stat["rot"].extend(fm_rot.cpu().tolist())
                stat["trans"].extend(fm_trans.cpu().tolist())
                stat["delta"].extend(torch.linalg.norm(_delta[:, :3].float(), dim=1).cpu().tolist())

    print(f"pos_dist: mean={np.mean(pos_all):.4f} median={np.median(pos_all):.4f}")
    for cm in args.dist_cm:
        stat = rank_stats[float(cm)]
        print(
            f"{cm:>5.1f}cm hard_neg: dist_med={np.median(stat['neg']):.4f} "
            f"gap_med={np.median(stat['gap']):+.4f} rank_acc={np.mean(stat['acc']) * 100:.1f}%"
        )
    if fm_stats and any(stat["trans"] for stat in fm_stats.values()):
        print(
            f"feature_metric: init_med={np.median(init_rot_all):.3f}deg/"
            f"{np.median(init_trans_all):.1f}mm"
        )
        for scale in args.fm_scales:
            stat = fm_stats[float(scale)]
            if not stat["trans"]:
                continue
            print(
                f"  scale={float(scale):+.2f}: fm_med={np.median(stat['rot']):.3f}deg/"
                f"{np.median(stat['trans']):.1f}mm  "
                f"delta_mean={np.mean(stat['delta']) * 1000.0:.1f}mm"
            )


if __name__ == "__main__":
    main()
