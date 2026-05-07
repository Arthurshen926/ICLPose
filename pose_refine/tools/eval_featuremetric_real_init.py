#!/usr/bin/env python3
"""Evaluate featuremetric Gauss-Newton alignment from real init poses.

Phase 0 validation: does LoFTR-PnP init + featuremetric GN produce a usable
baseline without any learned pose refinement?

Usage:
    CUDA_VISIBLE_DEVICES=0 python pose_refine/tools/eval_featuremetric_real_init.py \
        --config pose_refine/configs/concat_loc_cambridge_oldhospital_processed_adaptive_v7_locaware_matcher_v1.yaml \
        --init_cache result/result/feature_extract/pose_init_exports/oldhospital_netvlad_renderloftr_top10_test.npz \
        --max_iters 30 --damping 0.01 --verbose
"""
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
from pose_refine.utils.featuremetric import (
    compute_image_jacobian,
    compute_spatial_gradient,
)
from pose_refine.utils.lie_algebra import se3_exp


def camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def pose_errors(pose_pred: torch.Tensor, pose_gt: torch.Tensor):
    R_rel = torch.bmm(pose_pred[:, :3, :3].transpose(1, 2), pose_gt[:, :3, :3])
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    rot = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    trans = torch.norm(
        camera_centers_from_w2c(pose_pred) - camera_centers_from_w2c(pose_gt),
        dim=1,
    ) * 1000.0
    return rot * 180.0 / math.pi, trans


def scale_intrinsics(base_intr, orig_hw, target_hw):
    orig_h, orig_w = orig_hw
    h, w = target_hw
    return {
        "fx": float(base_intr["fx"] * w / orig_w),
        "fy": float(base_intr["fy"] * h / orig_h),
        "cx": float(base_intr["cx"] * w / orig_w),
        "cy": float(base_intr["cy"] * h / orig_h),
    }


@torch.no_grad()
def featuremetric_gn_step(
    query_feats: torch.Tensor,
    rendered_feats: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: dict,
    damping: float,
) -> torch.Tensor:
    """One Gauss-Newton step: minimize ||f_query - f_rendered||^2 over se(3)."""
    B, D, H, W = query_feats.shape
    N = H * W

    residual = query_feats - rendered_feats
    grad_u, grad_v = compute_spatial_gradient(rendered_feats)
    Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)

    gu = grad_u.reshape(B, D, N)
    gv = grad_v.reshape(B, D, N)
    r = residual.reshape(B, D, N)
    vf = valid.float()

    A = (gu * gu).sum(1) * vf
    B_ = (gv * gv).sum(1) * vf
    C = (gu * gv).sum(1) * vf
    Ru = (gu * r).sum(1) * vf
    Rv = (gv * r).sum(1) * vf

    JtJ = (
        torch.bmm((Ju * A.unsqueeze(-1)).transpose(1, 2), Ju)
        + torch.bmm((Ju * C.unsqueeze(-1)).transpose(1, 2), Jv)
        + torch.bmm((Jv * C.unsqueeze(-1)).transpose(1, 2), Ju)
        + torch.bmm((Jv * B_.unsqueeze(-1)).transpose(1, 2), Jv)
    )
    JtR = -(
        torch.bmm(Ju.transpose(1, 2), Ru.unsqueeze(-1))
        + torch.bmm(Jv.transpose(1, 2), Rv.unsqueeze(-1))
    )

    damping_mat = damping * torch.eye(6, device=JtJ.device).unsqueeze(0)
    try:
        L = torch.linalg.cholesky(JtJ + damping_mat)
        delta_xi = torch.cholesky_solve(JtR, L).squeeze(-1)
    except RuntimeError:
        delta_xi = torch.linalg.solve(JtJ + damping_mat, JtR).squeeze(-1)

    res_norm = (r**2).sum() / max(N * D, 1)
    return delta_xi, res_norm.item()


@torch.no_grad()
def run_featuremetric_alignment(
    runtime,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
    intrinsics: dict,
    query_fine: torch.Tensor,
    pose_init: torch.Tensor,
    max_iters: int = 30,
    damping_init: float = 0.01,
    use_coarse: bool = False,
    verbose: bool = False,
):
    """Run LM-style featuremetric alignment for a batch of queries."""
    B = pose_init.shape[0]
    device = pose_init.device
    current_pose = pose_init.clone()
    best_pose = current_pose.clone()
    best_residual = float("inf")
    damping = damping_init

    for k in range(max_iters):
        bundle = render_feature_bundle_batch(
            runtime.gaussians,
            runtime.renderer,
            runtime.refiner,
            current_pose,
            K,
            render_h,
            render_w,
            render_coarse=use_coarse,
        )
        rendered_fine = F.normalize(bundle["fine_features"].float(), dim=1)
        depth = bundle["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(1)

        q_norm = F.normalize(query_fine.float(), dim=1)

        delta_xi, res = featuremetric_gn_step(
            q_norm, rendered_fine, depth, intrinsics, damping
        )

        candidate_pose = se3_exp(delta_xi) @ current_pose

        # LM accept/reject: re-render at candidate and check residual
        bundle_cand = render_feature_bundle_batch(
            runtime.gaussians,
            runtime.renderer,
            runtime.refiner,
            candidate_pose,
            K,
            render_h,
            render_w,
            render_coarse=use_coarse,
        )
        rendered_cand = F.normalize(bundle_cand["fine_features"].float(), dim=1)
        cand_res = (q_norm - rendered_cand).pow(2).mean().item()

        if cand_res < res:
            current_pose = candidate_pose
            damping = max(damping / 3.0, 1e-6)
            if cand_res < best_residual:
                best_residual = cand_res
                best_pose = current_pose.clone()
            accepted = True
        else:
            damping = min(damping * 3.0, 1e4)
            accepted = False

        step_size = delta_xi.norm().item()
        if verbose and k < 5:
            status = "+" if accepted else "-"
            print(f"  [{k}] {status} res={cand_res:.6f} step={step_size:.4f} damp={damping:.1e}")

        if accepted and step_size < 1e-5:
            break

    return best_pose


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init_cache", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=30)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--use_coarse", action="store_true")
    parser.add_argument("--use_top_k", type=int, default=1, help="Use top-K candidates (1=top1 only)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build DCFF runtime
    print("Loading DCFF runtime...")
    runtime = build_dcff_runtime(config, device, printer=print)
    render_h = int(runtime.render_height)
    render_w = int(runtime.render_width)

    # Load dataset for GT poses and query features
    ds_cfg = config["dataset"]
    dataset = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        source_dir=ds_cfg.get("source_dir"),
        split="test",
        split_file=ds_cfg.get("test_split"),
        noise_rot_deg=0.0,
        noise_trans_m=0.0,
        coarse_hw=tuple(ds_cfg.get("coarse_hw", [render_h, render_w])),
        fine_hw=tuple(ds_cfg.get("fine_hw", [render_h, render_w])),
        cache_in_memory=False,
        normalize_features=ds_cfg.get("normalize_features", False),
    )

    # Intrinsics
    cameras = read_colmap_cameras(str(Path(ds_cfg["colmap_dir"]) / "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    render_intr = scale_intrinsics(
        dataset.intrinsics,
        (int(first_cam.height), int(first_cam.width)),
        (render_h, render_w),
    )
    K = intrinsics_to_K(render_intr, device)
    print(f"Render: {render_w}x{render_h}, intrinsics: {render_intr}")

    # Load init cache
    print(f"Loading init cache: {args.init_cache}")
    cache = np.load(args.init_cache, allow_pickle=True)
    pose_inits = torch.from_numpy(cache["pose_inits"]).float()
    query_stems = list(cache["query_image_stems"])
    init_sources = list(cache["init_sources"])
    if "pose_init_candidates" in cache:
        candidates = torch.from_numpy(cache["pose_init_candidates"]).float()
        candidate_valid = torch.from_numpy(cache["candidate_valid_mask"]).bool()
    else:
        candidates = None
        candidate_valid = None

    print(f"  {len(query_stems)} queries, sources: {set(init_sources)}")

    # Build name→index map for the dataset
    # Cache stems are like "seq8_frame00110", dataset names are "seq8/frame00110.png"
    ds_name_to_idx = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        name = sample["image_name"]  # e.g. "seq8/frame00110.png"
        stem = Path(name).stem  # "frame00110"
        parent = Path(name).parent.name  # "seq8"
        combined_stem = f"{parent}_{stem}"  # "seq8_frame00110"
        ds_name_to_idx[combined_stem] = i

    # Run evaluation
    init_rots, init_trans = [], []
    final_rots, final_trans = [], []
    skipped = 0
    num_samples = len(query_stems) if args.max_samples <= 0 else min(args.max_samples, len(query_stems))

    for qi in tqdm(range(num_samples), desc="featuremetric"):
        stem = query_stems[qi]
        if stem not in ds_name_to_idx:
            skipped += 1
            continue
        ds_idx = ds_name_to_idx[stem]
        sample = dataset[ds_idx]

        query_fine = sample["query_fine"].unsqueeze(0).to(device).float()
        pose_gt = sample["pose_gt"].unsqueeze(0).to(device).float()

        # Resize query features to render resolution if needed
        if query_fine.shape[-2:] != (render_h, render_w):
            query_fine = F.interpolate(query_fine, size=(render_h, render_w), mode="bilinear", align_corners=False)

        if args.use_top_k > 1 and candidates is not None:
            top_k_poses = candidates[qi, :args.use_top_k].to(device)
            valid_k = candidate_valid[qi, :args.use_top_k]
            best_res = float("inf")
            best_refined = None

            for ki in range(args.use_top_k):
                if not valid_k[ki]:
                    continue
                init_k = top_k_poses[ki:ki+1]
                refined_k = run_featuremetric_alignment(
                    runtime, K, render_h, render_w, render_intr,
                    query_fine, init_k,
                    max_iters=args.max_iters,
                    damping_init=args.damping,
                    use_coarse=args.use_coarse,
                    verbose=False,
                )
                # Pick by residual
                bundle_k = render_feature_bundle_batch(
                    runtime.gaussians, runtime.renderer, runtime.refiner,
                    refined_k, K, render_h, render_w, render_coarse=False,
                )
                r_feat = F.normalize(bundle_k["fine_features"].float(), dim=1)
                q_norm = F.normalize(query_fine, dim=1)
                res_k = (q_norm - r_feat).pow(2).mean().item()
                if res_k < best_res:
                    best_res = res_k
                    best_refined = refined_k

            refined_pose = best_refined if best_refined is not None else pose_inits[qi:qi+1].to(device)
        else:
            init_pose = pose_inits[qi:qi+1].to(device)
            refined_pose = run_featuremetric_alignment(
                runtime, K, render_h, render_w, render_intr,
                query_fine, init_pose,
                max_iters=args.max_iters,
                damping_init=args.damping,
                use_coarse=args.use_coarse,
                verbose=args.verbose and qi < 3,
            )

        init_rot, init_t = pose_errors(pose_inits[qi:qi+1].to(device), pose_gt)
        final_rot, final_t = pose_errors(refined_pose, pose_gt)
        init_rots.append(init_rot.item())
        init_trans.append(init_t.item())
        final_rots.append(final_rot.item())
        final_trans.append(final_t.item())

    if skipped > 0:
        print(f"Warning: skipped {skipped} queries (not found in dataset)")

    init_rots = np.array(init_rots)
    init_trans = np.array(init_trans)
    final_rots = np.array(final_rots)
    final_trans = np.array(final_trans)

    print(f"\n{'='*60}")
    print(f"Featuremetric GN alignment (max_iters={args.max_iters}, damping={args.damping})")
    print(f"Init cache: {args.init_cache}")
    print(f"Samples: {len(init_rots)}")
    print(f"{'='*60}")
    print(f"  Init:    {np.median(init_rots):.3f} deg / {np.median(init_trans):.1f} mm")
    print(f"  Refined: {np.median(final_rots):.3f} deg / {np.median(final_trans):.1f} mm")
    print(f"  Mean:    {np.mean(final_rots):.3f} deg / {np.mean(final_trans):.1f} mm")

    recall_thresholds = [
        (1.0, 50.0), (1.0, 100.0), (2.0, 100.0), (5.0, 250.0),
    ]
    for r_thresh, t_thresh in recall_thresholds:
        r = np.mean((final_rots < r_thresh) & (final_trans < t_thresh)) * 100.0
        print(f"  Recall@{r_thresh}deg/{t_thresh:.0f}mm: {r:.1f}%")

    # Per-sample gain
    gain_rot = init_rots - final_rots
    gain_trans = init_trans - final_trans
    improved = np.sum(final_trans < init_trans)
    print(f"\n  Gain median: {np.median(gain_rot):.3f} deg / {np.median(gain_trans):.1f} mm")
    print(f"  Improved: {improved}/{len(final_trans)} ({100*improved/len(final_trans):.1f}%)")


if __name__ == "__main__":
    main()
