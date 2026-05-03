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
from feature_extract.students.radio_query_student import DepthAwareLocalMatcher
from pose_refine.models.concat_pose_net import (
    ConcatPoseNet,
    local_correlation,
    soft_argmax_flow_from_correlation,
)
from pose_refine.runtime import build_concat_pose_model
from pose_refine.train_impl import (
    apply_pose_delta,
    feature_metric_pose_update,
    masked_feature_cosine_distance_per_sample,
    perturb_w2c_camera_center,
)
from pose_refine.utils.geometry_solver import compute_image_jacobian, diff_pose_solve


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
    if mode in {"shared", "shared_linear"}:
        query = model.proj_shared(query)
        rendered = model.proj_shared(rendered)
    elif mode == "identity":
        pass
    else:
        query = model.proj_query(query)
        rendered = model.proj_render(rendered)
    return F.normalize(query.float(), dim=1), F.normalize(rendered.float(), dim=1)


def load_query_local_matcher(
    config_path: str,
    checkpoint_path: str,
    device: torch.device | str,
) -> DepthAwareLocalMatcher:
    """Load only the query-student local matcher for pose diagnostics."""
    with open(config_path, "r", encoding="utf-8") as f:
        query_config = yaml.safe_load(f)
    model_cfg = query_config.get("model", {})
    if not bool(model_cfg.get("local_matcher_enabled", False)):
        raise ValueError(f"Query config does not enable local matcher: {config_path}")
    matcher = DepthAwareLocalMatcher(
        radius=int(model_cfg.get("local_matcher_radius", 4)),
        hidden_dim=int(model_cfg.get("local_matcher_hidden_dim", 64)),
        zero_init=False,
        residual_scale=float(model_cfg.get("local_matcher_residual_scale", 1.0)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    matcher_state = {
        key[len("local_matcher."):]: value
        for key, value in state.items()
        if key.startswith("local_matcher.")
    }
    if not matcher_state:
        raise RuntimeError(f"No local_matcher.* weights found in {checkpoint_path}")
    matcher.load_state_dict(matcher_state, strict=True)
    matcher.to(device)
    matcher.eval()
    return matcher


def correlation_wls_pose_update(
    query_feat: torch.Tensor,
    rendered_feat: torch.Tensor,
    depth: torch.Tensor,
    pose_ref: torch.Tensor,
    pose_gt: torch.Tensor,
    intrinsics: dict,
    *,
    radius: int,
    temperature: float,
    damping: float,
    valid_mask=None,
    matcher=None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Local feature correlation -> soft subpixel flow -> depth WLS update."""
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        if query.shape[-2:] != rendered.shape[-2:]:
            query = F.interpolate(query, rendered.shape[-2:], mode="bilinear", align_corners=False)
        query_n = F.normalize(query, dim=1)
        rendered_n = F.normalize(rendered, dim=1)
        corr = local_correlation(rendered_n, query_n, radius=int(radius)).float()
        if matcher is not None:
            corr = matcher(corr, depth=depth, valid_mask=valid_mask).float()
        probs = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
        flow = soft_argmax_flow_from_correlation(
            corr,
            radius=int(radius),
            temperature=float(temperature),
        ).float()
        conf = probs.max(dim=1, keepdim=True).values
        depth_s = depth.float()
        if depth_s.ndim == 4:
            depth_s = depth_s.squeeze(1)
        if valid_mask is None:
            valid_w = (depth_s > 0.05).unsqueeze(1).float()
        elif valid_mask.ndim == 3:
            valid_w = valid_mask.unsqueeze(1).float()
        else:
            valid_w = valid_mask.float()
        if valid_w.shape[-2:] != rendered.shape[-2:]:
            valid_w = F.interpolate(valid_w, rendered.shape[-2:], mode="nearest")

        gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
            pose_ref.float(),
            pose_gt.float(),
            depth_s,
            rendered.shape[-2:],
            intrinsics,
        )
        in_window = (
            (gt_valid > 0.5)
            & (valid_w > 0.5)
            & (gt_flow[:, :1] >= -radius)
            & (gt_flow[:, :1] <= radius)
            & (gt_flow[:, 1:2] >= -radius)
            & (gt_flow[:, 1:2] <= radius)
        ).float()
        epe_map = torch.linalg.norm(flow - gt_flow.float(), dim=1, keepdim=True)
        denom = in_window.sum().clamp(min=1.0)
        epe = (epe_map * in_window).sum() / denom
        cov = in_window.mean()
        conf_w = conf * in_window

        Ju, Jv, depth_valid = compute_image_jacobian(depth_s, intrinsics)
        delta_xi = diff_pose_solve(
            flow,
            conf_w.expand(-1, 2, -1, -1).contiguous(),
            Ju,
            Jv,
            depth_valid,
            damping=float(damping),
        )
        pose_pred = apply_pose_delta(pose_ref.float(), delta_xi.float())
        metrics = {
            "flow_epe": epe.detach(),
            "flow_cov": cov.detach(),
            "conf_mean": ((conf * in_window).sum() / denom).detach(),
            "delta_trans_m": torch.linalg.norm(delta_xi[:, :3], dim=1).detach(),
        }
        return delta_xi, pose_pred, metrics


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
    parser.add_argument("--corr_radius", type=int, default=8)
    parser.add_argument("--corr_temperature", type=float, default=0.04)
    parser.add_argument("--corr_wls_damping", type=float, default=1e-3)
    parser.add_argument("--no_corr_wls", action="store_true")
    parser.add_argument("--pose_checkpoint", default=None)
    parser.add_argument("--query_student_config", default=None)
    parser.add_argument("--query_student_checkpoint", default=None)
    parser.add_argument("--use_projection", action="store_true")
    parser.add_argument("--no_feature_metric", action="store_true")
    parser.add_argument(
        "--query_source",
        choices=["dataset", "self_render"],
        default="dataset",
        help="Use exported query features or GT rendered DCFF features as the query signal.",
    )
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
    query_matcher = None
    if args.query_student_config or args.query_student_checkpoint:
        if not args.query_student_config or not args.query_student_checkpoint:
            raise ValueError("--query_student_config and --query_student_checkpoint must be provided together")
        query_matcher = load_query_local_matcher(
            args.query_student_config,
            args.query_student_checkpoint,
            device,
        )
        print(
            "Loaded query local matcher: "
            f"config={args.query_student_config} checkpoint={args.query_student_checkpoint}"
        )

    pos_all = []
    rank_stats = {float(cm): {"neg": [], "acc": [], "gap": []} for cm in args.dist_cm}
    init_rot_all, init_trans_all = [], []
    fm_stats = {float(scale): {"rot": [], "trans": [], "delta": []} for scale in args.fm_scales}
    corr_stats = {"rot": [], "trans": [], "epe": [], "cov": [], "conf": [], "delta": []}

    print(
        f"Render: {render_w}x{render_h}  split={args.split}  "
        f"noise={args.noise_deg}deg/{args.noise_m}m  frame={args.frame}  "
        f"query_source={args.query_source}"
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc="dcff-cm", leave=False)):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        query_fine = batch["query_fine"].to(device).float()
        pose_gt = batch["pose_gt"].to(device).float()
        pose_init = batch["pose_init"].to(device).float()

        gt_bundle = render_bundle(runtime, pose_gt, K, render_h, render_w)
        gt_mask = (gt_bundle["depth"].float() > 0.05).unsqueeze(1)
        query_eval = gt_bundle["fine_features"].float() if args.query_source == "self_render" else query_fine
        pos_dist = masked_feature_cosine_distance_per_sample(
            query_eval,
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
                        query_eval,
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
            fm_query = query_eval
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
        if not args.no_corr_wls:
            init_bundle = render_bundle(runtime, pose_init, K, render_h, render_w)
            init_mask = (init_bundle["depth"].float() > 0.05).unsqueeze(1)
            corr_query = query_eval
            corr_rendered = init_bundle["fine_features"].float()
            if proj_model is not None:
                corr_query, corr_rendered = project_feature_pair(
                    proj_model,
                    config,
                    query_fine,
                    corr_rendered,
                )
            _corr_delta, corr_pose, corr_metrics = correlation_wls_pose_update(
                corr_query,
                corr_rendered,
                init_bundle["depth"].float(),
                pose_init,
                pose_gt,
                render_intr,
                radius=args.corr_radius,
                temperature=args.corr_temperature,
                damping=args.corr_wls_damping,
                valid_mask=init_mask,
                matcher=query_matcher,
            )
            corr_rot, corr_trans = pose_errors(corr_pose, pose_gt)
            corr_stats["rot"].extend(corr_rot.cpu().tolist())
            corr_stats["trans"].extend(corr_trans.cpu().tolist())
            corr_stats["epe"].append(float(corr_metrics["flow_epe"].cpu()))
            corr_stats["cov"].append(float(corr_metrics["flow_cov"].cpu()))
            corr_stats["conf"].append(float(corr_metrics["conf_mean"].cpu()))
            corr_stats["delta"].extend(corr_metrics["delta_trans_m"].cpu().tolist())

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
    if corr_stats["trans"]:
        print(
            f"corr_wls: r={args.corr_radius} temp={args.corr_temperature:.3f} "
            f"corr_med={np.median(corr_stats['rot']):.3f}deg/"
            f"{np.median(corr_stats['trans']):.1f}mm  "
            f"flow_epe={np.mean(corr_stats['epe']):.3f}px "
            f"cov={np.mean(corr_stats['cov']):.3f} "
            f"conf={np.mean(corr_stats['conf']):.3f} "
            f"delta_mean={np.mean(corr_stats['delta']) * 1000.0:.1f}mm"
        )


if __name__ == "__main__":
    main()
