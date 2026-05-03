#!/usr/bin/env python3
"""Evaluate the concat localizer under retrieval-based real pose init."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.radio_loc_dataset import (  # noqa: E402
    camera_params_to_intrinsics,
    collate_fn,
    read_colmap_cameras,
)
from data.radio_loc_retrieval_dataset import RadioLocRetrievalDataset  # noqa: E402
from pose_refine import (  # noqa: E402
    load_concat_pose_checkpoint,
    load_concat_pose_model as load_model,
    run_model_refine_iteration,
)
from pose_refine.utils.geometry_solver import feature_metric_solve, pnp_ransac_solve  # noqa: E402
from pose_refine.utils.lie_algebra import se3_exp  # noqa: E402
from feature_field import build_dcff, intrinsics_to_K, render_batch, render_feature_bundle_batch  # noqa: E402
from feature_field.runtime import apply_localization_map_state  # noqa: E402
from feature_field.utils.loc_reporting import save_experiment_bundle  # noqa: E402
from feature_field.utils.project_config import load_mainline_config  # noqa: E402
from feature_field.utils.project_paths import resolve_repo_path  # noqa: E402
from feature_field.utils.real_init_vis import (  # noqa: E402
    feature_map_to_rgb,
    load_rgb_image,
    save_triptych,
)


def compute_pose_errors(pose_pred: torch.Tensor, pose_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.amp.autocast(enabled=False):
        pose_pred = pose_pred.float()
        pose_gt = pose_gt.float()
        R_pred = pose_pred[:, :3, :3]
        R_gt = pose_gt[:, :3, :3]
        R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        rot_err = torch.acos(cos_angle) * 180.0 / math.pi
        c_pred = camera_centers_from_w2c(pose_pred)
        c_gt = camera_centers_from_w2c(pose_gt)
        trans_err = torch.norm(c_pred - c_gt, dim=1) * 1000.0
    return rot_err, trans_err


def masked_mean_2d(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask.float()
    denom = mask.float().sum(dim=(-1, -2)).clamp(min=1.0)
    return masked.sum(dim=(-1, -2)) / denom


def masked_mean_std_2d(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = masked_mean_2d(values, mask)
    centered = values - mean[:, None, None]
    var = masked_mean_2d(centered.pow(2), mask)
    return mean, torch.sqrt(var.clamp(min=1e-8))


def summarize_pose_metrics(rot_errs: np.ndarray, trans_errs: np.ndarray) -> Dict[str, float]:
    return {
        "rot_mean": float(np.nanmean(rot_errs)),
        "rot_median": float(np.nanmedian(rot_errs)),
        "trans_mean": float(np.nanmean(trans_errs)),
        "trans_median": float(np.nanmedian(trans_errs)),
        "pct_1deg": float(np.mean(rot_errs < 1.0) * 100.0),
        "pct_5deg": float(np.mean(rot_errs < 5.0) * 100.0),
        "pct_10deg": float(np.mean(rot_errs < 10.0) * 100.0),
        "pct_1000mm": float(np.mean(trans_errs < 1000.0) * 100.0),
        "pct_2000mm": float(np.mean(trans_errs < 2000.0) * 100.0),
        "joint_01deg_53mm": float(np.mean((rot_errs < 0.1) & (trans_errs < 5.3)) * 100.0),
        "joint_1deg_50mm": float(np.mean((rot_errs < 1.0) & (trans_errs < 50.0)) * 100.0),
        "joint_1deg_100mm": float(np.mean((rot_errs < 1.0) & (trans_errs < 100.0)) * 100.0),
        "joint_2deg_100mm": float(np.mean((rot_errs < 2.0) & (trans_errs < 100.0)) * 100.0),
        "joint_5deg_250mm": float(np.mean((rot_errs < 5.0) & (trans_errs < 250.0)) * 100.0),
        "joint_5deg_100mm": float(np.mean((rot_errs < 5.0) & (trans_errs < 100.0)) * 100.0),
        "joint_5deg_1000mm": float(np.mean((rot_errs < 5.0) & (trans_errs < 1000.0)) * 100.0),
        "joint_10deg_2000mm": float(np.mean((rot_errs < 10.0) & (trans_errs < 2000.0)) * 100.0),
    }


def summarize_full_pipeline_metrics(
    *,
    init_rot_errs,
    init_trans_errs,
    final_rot_errs,
    final_trans_errs,
) -> Dict[str, Dict[str, float]]:
    """Summarize the single paper-facing real-init full-pipeline protocol."""
    init_rot = np.asarray(init_rot_errs, dtype=np.float64)
    init_trans = np.asarray(init_trans_errs, dtype=np.float64)
    final_rot = np.asarray(final_rot_errs, dtype=np.float64)
    final_trans = np.asarray(final_trans_errs, dtype=np.float64)
    init = summarize_pose_metrics(init_rot, init_trans)
    final = summarize_pose_metrics(final_rot, final_trans)
    return {
        "protocol": "real_init_full_pipeline",
        "init": init,
        "final": final,
        "gain": {
            "rot_median_deg": float(init["rot_median"] - final["rot_median"]),
            "trans_median_mm": float(init["trans_median"] - final["trans_median"]),
            "joint_1deg_50mm": float(final["joint_1deg_50mm"] - init["joint_1deg_50mm"]),
            "joint_1deg_100mm": float(final["joint_1deg_100mm"] - init["joint_1deg_100mm"]),
            "joint_5deg_250mm": float(final["joint_5deg_250mm"] - init["joint_5deg_250mm"]),
        },
    }


def infer_retrieval_feature_dir(config: Dict, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    ds_cfg = config.get("dataset", {})
    scene_hint = None
    if ds_cfg.get("source_dir"):
        scene_hint = os.path.basename(os.path.normpath(ds_cfg["source_dir"]))
    if scene_hint:
        candidate = resolve_repo_path(Path("output") / "features_multiscale_compressed" / scene_hint)
        if candidate and candidate.is_dir():
            return str(candidate)
    return None


def invert_pose_w2c(pose_w2c: torch.Tensor) -> torch.Tensor:
    R = pose_w2c[:, :3, :3]
    t = pose_w2c[:, :3, 3:4]
    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t)
    pose_c2w = torch.eye(4, device=pose_w2c.device, dtype=pose_w2c.dtype).unsqueeze(0).repeat(
        pose_w2c.shape[0], 1, 1
    )
    pose_c2w[:, :3, :3] = R_inv
    pose_c2w[:, :3, 3] = t_inv.squeeze(-1)
    return pose_c2w


def project_rotation_to_so3(rotation: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(rotation)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone()
        U[:, -1] *= -1.0
        R = U @ Vh
    return R


def camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    pose_c2w = invert_pose_w2c(poses_w2c)
    return pose_c2w[:, :3, 3]


def fuse_pose_centroid(poses_w2c: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if poses_w2c.shape[0] == 1:
        return poses_w2c[0]

    pose_c2w = invert_pose_w2c(poses_w2c)
    centers = pose_c2w[:, :3, 3]
    rotations = pose_c2w[:, :3, :3]

    if weights is None:
        weights = torch.ones((poses_w2c.shape[0],), device=poses_w2c.device, dtype=poses_w2c.dtype)
    weights = torch.clamp(weights.float(), min=0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    center_mean = torch.sum(weights[:, None] * centers, dim=0)
    rotation_mean = torch.sum(weights[:, None, None] * rotations, dim=0)
    rotation_proj = project_rotation_to_so3(rotation_mean)

    pose_c2w_fused = torch.eye(4, device=poses_w2c.device, dtype=poses_w2c.dtype)
    pose_c2w_fused[:3, :3] = rotation_proj
    pose_c2w_fused[:3, 3] = center_mean
    return invert_pose_w2c(pose_c2w_fused.unsqueeze(0))[0]


def fuse_candidate_poses(
    poses_w2c: torch.Tensor,
    method: str,
    retrieval_scores: Optional[torch.Tensor] = None,
    consensus_radius_m: float = 1.0,
    consensus_min_size: int = 2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    num_hyp = int(poses_w2c.shape[0])
    if num_hyp == 0:
        raise ValueError("Cannot fuse zero poses.")
    if num_hyp == 1 or method == "none":
        return poses_w2c[0], {
            "cluster_size": 1.0,
            "num_hypotheses": float(num_hyp),
            "anchor_idx": 0.0,
            "selected_idx": 0.0,
            "used_top1_fallback": 1.0 if method == "none" or num_hyp == 1 else 0.0,
        }
    if method == "centroid":
        return fuse_pose_centroid(poses_w2c), {
            "cluster_size": float(num_hyp),
            "num_hypotheses": float(num_hyp),
            "anchor_idx": 0.0,
            "selected_idx": 0.0,
            "used_top1_fallback": 0.0,
        }

    centers = camera_centers_from_w2c(poses_w2c)
    pairwise = torch.cdist(centers, centers)
    support = (pairwise <= float(consensus_radius_m)).sum(dim=1).float()

    tie_break = torch.zeros_like(support)
    if retrieval_scores is not None and retrieval_scores.numel() == num_hyp:
        retrieval_scores = retrieval_scores.float()
        if float(retrieval_scores.abs().sum().item()) > 0:
            tie_break = (retrieval_scores - retrieval_scores.min()) / (
                retrieval_scores.max() - retrieval_scores.min() + 1e-8
            )
    anchor_idx = int(torch.argmax(support + 1e-3 * tie_break).item())
    cluster_mask = pairwise[anchor_idx] <= float(consensus_radius_m)
    cluster_size = int(cluster_mask.sum().item())
    if cluster_size < int(consensus_min_size):
        return poses_w2c[0], {
            "cluster_size": float(cluster_size),
            "num_hypotheses": float(num_hyp),
            "anchor_idx": float(anchor_idx),
            "selected_idx": 0.0,
            "used_top1_fallback": 1.0,
        }

    if method == "consensus_centroid":
        fused_pose = fuse_pose_centroid(poses_w2c[cluster_mask])
        selected_idx = anchor_idx
    else:
        fused_pose = poses_w2c[anchor_idx]
        selected_idx = anchor_idx
    return fused_pose, {
        "cluster_size": float(cluster_size),
        "num_hypotheses": float(num_hyp),
        "anchor_idx": float(anchor_idx),
        "selected_idx": float(selected_idx),
        "used_top1_fallback": 0.0,
    }


def refine_pose_batch(
    model,
    gaussians,
    dcff_renderer,
    feat_sharp,
    query_fine: torch.Tensor,
    query_coarse: Optional[torch.Tensor],
    pose_init: torch.Tensor,
    K: torch.Tensor,
    render_intr: Dict,
    outer_iters: int,
    solver: str,
    irls_iters: int,
    robust_kernel: str,
    direct_refine_iters: int,
    collect_diagnostics: bool = False,
):
    pose_cur = pose_init.clone()
    use_flow = not solver.startswith("direct")
    n_flow_iters = outer_iters if use_flow else 0
    batch_size = pose_init.shape[0]
    zero_stat = torch.zeros(batch_size, device=pose_init.device, dtype=torch.float32)
    diag_acc = {
        "flow_mag_mean": [],
        "flow_mag_std": [],
        "flow_mag_max": [],
        "confidence_mean": [],
        "confidence_std": [],
        "confidence_lowfrac": [],
        "flow_step_mean": [],
        "delta_xi_norm": [],
    }
    last_depth = None

    def render_bundle_fn(poses_w2c: torch.Tensor, render_coarse: bool | None) -> Dict[str, Optional[torch.Tensor]]:
        return render_feature_bundle_batch(
            gaussians,
            dcff_renderer,
            feat_sharp,
            poses_w2c,
            K,
            *query_fine.shape[-2:],
            render_coarse=render_coarse,
        )

    for outer_i in range(n_flow_iters):
        fwd_irls = irls_iters if solver == "wls" else 0
        fwd_kernel = robust_kernel if solver == "wls" else None
        iter_state = run_model_refine_iteration(
            model,
            render_bundle_fn,
            query_fine,
            query_coarse,
            pose_cur,
            render_intr,
            outer_iter=outer_i,
            irls_iters=fwd_irls,
            robust_kernel=fwd_kernel,
            autocast_enabled=True,
            apply_fine_update=False,
        )
        pred = iter_state["fine_pred"]
        fine_bundle = iter_state["fine_bundle"]
        depth = fine_bundle["depth"]
        pose_mid = iter_state["pose_mid"]
        last_depth = depth

        use_pnp = (solver == "pnp") or (solver == "hybrid" and outer_i == n_flow_iters - 1)
        if use_pnp and "flow" in pred:
            with torch.cuda.amp.autocast(enabled=False):
                flow_f = pred["flow"].detach().float()
                conf_f = pred["confidence"].detach().float()
                depth_for_pnp = depth.detach().float()
                if depth_for_pnp.dim() == 3:
                    depth_for_pnp = depth_for_pnp.unsqueeze(1)
                pnp_xi = pnp_ransac_solve(
                    flow_f,
                    depth_for_pnp,
                    render_intr,
                    confidence=conf_f,
                    reprojection_threshold=4.0,
                    n_iters=2000,
                )
                T_delta = se3_exp(pnp_xi.float())
        elif solver == "wls_full" and "delta_xi_full" in pred:
            with torch.cuda.amp.autocast(enabled=False):
                T_delta = se3_exp(pred["delta_xi_full"].float())
        elif "delta_xi" in pred:
            with torch.cuda.amp.autocast(enabled=False):
                T_delta = se3_exp(pred["delta_xi"].float())
        else:
            continue

        with torch.cuda.amp.autocast(enabled=False):
            pose_cur = torch.bmm(T_delta, pose_mid.float())

        if collect_diagnostics:
            depth_mask = depth > 0.05
            flow_map = pred["flow"].float()
            conf_map = pred["confidence"].float().mean(dim=1)
            flow_mag = torch.norm(flow_map, dim=1)
            flow_mean, flow_std = masked_mean_std_2d(flow_mag, depth_mask)
            conf_mean, conf_std = masked_mean_std_2d(conf_map, depth_mask)
            denom = depth_mask.float().sum(dim=(-1, -2)).clamp(min=1.0)
            conf_lowfrac = ((conf_map < 0.25) & depth_mask).float().sum(dim=(-1, -2)) / denom
            flow_max = flow_mag.masked_fill(~depth_mask, 0.0).amax(dim=(-1, -2))

            flow_preds = [fp.float() for fp in pred.get("flow_preds", [])]
            if len(flow_preds) > 1:
                step_stats = []
                prev = flow_preds[0]
                for cur in flow_preds[1:]:
                    step_mag = torch.norm(cur - prev, dim=1)
                    step_stats.append(masked_mean_2d(step_mag, depth_mask))
                    prev = cur
                flow_step_mean = torch.stack(step_stats, dim=0).mean(dim=0)
            else:
                flow_step_mean = zero_stat

            delta_xi = pred.get("delta_xi")
            delta_xi_norm = delta_xi.float().norm(dim=1) if delta_xi is not None else zero_stat

            diag_acc["flow_mag_mean"].append(flow_mean)
            diag_acc["flow_mag_std"].append(flow_std)
            diag_acc["flow_mag_max"].append(flow_max)
            diag_acc["confidence_mean"].append(conf_mean)
            diag_acc["confidence_std"].append(conf_std)
            diag_acc["confidence_lowfrac"].append(conf_lowfrac)
            diag_acc["flow_step_mean"].append(flow_step_mean)
            diag_acc["delta_xi_norm"].append(delta_xi_norm)

    n_direct = direct_refine_iters if solver != "direct" else outer_iters
    for _ in range(n_direct):
        ref_fine, depth = render_batch(gaussians, dcff_renderer, feat_sharp, pose_cur, K, *query_fine.shape[-2:])
        last_depth = depth
        with torch.cuda.amp.autocast(enabled=False):
            depth_s = depth.float() if depth.dim() == 3 else depth.squeeze(1).float()
            damping = 1.0 if "highdamp" in solver else 1e-2
            direct_xi, _ = feature_metric_solve(
                query_fine.float(), ref_fine.float(), depth_s, render_intr, damping=damping
            )
            if "neg" in solver:
                direct_xi = -direct_xi
            T_delta = se3_exp(direct_xi.float())
            pose_cur = torch.bmm(T_delta, pose_cur.float())

    final_render, final_depth = render_batch(gaussians, dcff_renderer, feat_sharp, pose_cur, K, *query_fine.shape[-2:])
    if last_depth is None:
        last_depth = final_depth
    diagnostics = {}
    for key, values in diag_acc.items():
        diagnostics[key] = torch.stack(values, dim=0).mean(dim=0) if values else zero_stat
    diagnostics["depth_valid_ratio"] = (last_depth > 0.05).float().mean(dim=(-1, -2))
    if collect_diagnostics:
        return pose_cur, final_render, diagnostics
    return pose_cur, final_render


@torch.no_grad()
def render_rgb_alpha_batch(
    gaussians,
    dcff_renderer,
    poses_w2c: torch.Tensor,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb_list = []
    alpha_list = []
    for i in range(poses_w2c.shape[0]):
        result = dcff_renderer(
            gaussians,
            viewmat=poses_w2c[i].float(),
            K=K,
            width=render_w,
            height=render_h,
            render_coarse=False,
        )
        rgb_list.append(result["rgb"].squeeze(0).float())
        alpha_list.append(result["alpha"].squeeze(0).float())
    return torch.stack(rgb_list, dim=0), torch.stack(alpha_list, dim=0)


def cascade_refine_single(
    model,
    gaussians,
    dcff_renderer,
    feat_sharp,
    query_fine: torch.Tensor,
    query_coarse: Optional[torch.Tensor],
    best_pose: torch.Tensor,
    K: torch.Tensor,
    render_intr: Dict,
    outer_iters: int,
    solver: str,
    irls_iters: int,
    robust_kernel: str,
    direct_refine_iters: int,
    cascade_starts: int,
    cascade_noise_deg: float,
    cascade_noise_m: float,
    render_h: int,
    render_w: int,
) -> torch.Tensor:
    """Run one cascade round: perturb best_pose, multi-start refine, consensus select."""
    from data.radio_loc_dataset import add_pose_noise

    best_np = best_pose.detach().cpu().squeeze(0).numpy()
    start_poses = []
    for _ in range(cascade_starts):
        perturbed = add_pose_noise(best_np, cascade_noise_deg, cascade_noise_m)
        start_poses.append(torch.from_numpy(perturbed).float().unsqueeze(0))
    start_poses_t = torch.cat(start_poses, dim=0).to(best_pose.device)

    query_fine_rep = query_fine.expand(cascade_starts, -1, -1, -1)
    query_coarse_rep = query_coarse.expand(cascade_starts, -1, -1, -1) if query_coarse is not None else None

    refined_poses, _ = refine_pose_batch(
        model=model,
        gaussians=gaussians,
        dcff_renderer=dcff_renderer,
        feat_sharp=feat_sharp,
        query_fine=query_fine_rep,
        query_coarse=query_coarse_rep,
        pose_init=start_poses_t,
        K=K,
        render_intr=render_intr,
        outer_iters=outer_iters,
        solver=solver,
        irls_iters=irls_iters,
        robust_kernel=robust_kernel,
        direct_refine_iters=direct_refine_iters,
    )

    # Consensus selection: pick pose closest to median translation
    all_t = refined_poses[:, :3, 3]
    median_t = all_t.median(dim=0).values
    dists = (all_t - median_t.unsqueeze(0)).norm(dim=1)
    best_idx = int(dists.argmin().item())
    return refined_poses[best_idx : best_idx + 1]


@torch.no_grad()
def evaluate_real_init(
    model,
    gaussians,
    dcff_renderer,
    feat_sharp,
    val_loader,
    device,
    outer_iters: int,
    gru_iters: int,
    render_h: int,
    render_w: int,
    use_coarse: bool,
    solver: str,
    irls_iters: int,
    robust_kernel: str,
    direct_refine_iters: int,
    qual_dir: str,
    qual_limit: int,
    retrieval_topk: int = 1,
    pose_fusion: str = "none",
    consensus_radius_m: float = 1.0,
    consensus_min_size: int = 2,
    cascade_rounds: int = 0,
    cascade_starts: int = 10,
    cascade_noise_deg: float = 0.3,
    cascade_noise_m: float = 0.05,
) -> Tuple[Dict, List[Dict]]:
    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = gru_iters

    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    init_rot_all: List[float] = []
    init_trans_all: List[float] = []
    top1_final_rot_all: List[float] = []
    top1_final_trans_all: List[float] = []
    fused_final_rot_all: List[float] = []
    fused_final_trans_all: List[float] = []
    oracle_final_rot_all: List[float] = []
    oracle_final_trans_all: List[float] = []
    cluster_sizes: List[float] = []
    used_top1_fallback: List[float] = []
    records: List[Dict] = []
    qual_saved = 0

    for batch in tqdm(val_loader, desc="real-init eval", leave=False):
        query_fine = batch["query_fine"].to(device)
        query_coarse = batch.get("query_coarse")
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.to(device)
        else:
            query_coarse = None
        pose_gt = batch["pose_gt"].to(device)
        pose_init = batch["pose_init"].to(device)
        pose_init_candidates = batch.get("pose_init_candidates")
        candidate_valid_mask = batch.get("candidate_valid_mask")
        retrieval_scores_candidates = batch.get("retrieval_scores_candidates")
        retrieval_frame_ids_candidates = batch.get("retrieval_frame_ids_candidates")
        retrieval_image_names_candidates = batch.get("retrieval_image_names_candidates")

        batch_size = pose_gt.shape[0]
        for i in range(batch_size):
            query_fine_i = query_fine[i : i + 1]
            query_coarse_i = query_coarse[i : i + 1] if query_coarse is not None else None
            pose_gt_i = pose_gt[i : i + 1]

            if pose_init_candidates is not None and retrieval_topk > 1:
                valid_mask = candidate_valid_mask[i]
                candidate_poses = pose_init_candidates[i][valid_mask].to(device)
                candidate_scores = retrieval_scores_candidates[i][valid_mask].to(device)
                candidate_frame_ids = retrieval_frame_ids_candidates[i][valid_mask].cpu().tolist()
                candidate_image_names = [
                    retrieval_image_names_candidates[i][j]
                    for j, keep in enumerate(valid_mask.cpu().tolist())
                    if keep
                ]
            else:
                candidate_poses = pose_init[i : i + 1]
                candidate_scores = torch.tensor(
                    [float(batch["retrieval_score"][i])], device=device, dtype=torch.float32
                )
                candidate_frame_ids = [int(batch["retrieval_frame_id"][i])]
                candidate_image_names = [batch["retrieval_image_name"][i]]

            hyp_count = int(candidate_poses.shape[0])
            query_fine_rep = query_fine_i.repeat(hyp_count, 1, 1, 1)
            query_coarse_rep = query_coarse_i.repeat(hyp_count, 1, 1, 1) if query_coarse_i is not None else None
            pose_gt_rep = pose_gt_i.repeat(hyp_count, 1, 1)

            init_rot_h, init_trans_h = compute_pose_errors(candidate_poses, pose_gt_rep)
            init_render_top1, _ = render_batch(
                gaussians, dcff_renderer, feat_sharp, candidate_poses[:1], K, render_h, render_w
            )
            refined_poses, refined_renders = refine_pose_batch(
                model=model,
                gaussians=gaussians,
                dcff_renderer=dcff_renderer,
                feat_sharp=feat_sharp,
                query_fine=query_fine_rep,
                query_coarse=query_coarse_rep,
                pose_init=candidate_poses,
                K=K,
                render_intr=render_intr,
                outer_iters=outer_iters,
                solver=solver,
                irls_iters=irls_iters,
                robust_kernel=robust_kernel,
                direct_refine_iters=direct_refine_iters,
            )
            final_rot_h, final_trans_h = compute_pose_errors(refined_poses, pose_gt_rep)

            top1_final_pose = refined_poses[:1]
            top1_final_render = refined_renders[:1]
            if pose_fusion == "rgb_select":
                rgb_path = batch["query_rgb_path"][i]
                if rgb_path and os.path.isfile(rgb_path):
                    query_rgb_np = load_rgb_image(rgb_path, target_hw=(render_h, render_w))
                    query_rgb_t = (
                        torch.from_numpy(np.array(query_rgb_np, copy=True))
                        .to(device=device, dtype=torch.float32)
                        .permute(2, 0, 1)
                        / 255.0
                    )
                    candidate_rgb, candidate_alpha = render_rgb_alpha_batch(
                        gaussians, dcff_renderer, refined_poses, K, render_h, render_w
                    )
                    rgb_errors = []
                    for h in range(candidate_rgb.shape[0]):
                        valid = candidate_alpha[h] > 0.5
                        if valid.any():
                            diff = (candidate_rgb[h] - query_rgb_t).pow(2)
                            rgb_errors.append(float(diff[valid.expand_as(diff)].mean().item()))
                        else:
                            rgb_errors.append(float("inf"))
                    selected_idx = int(np.argmin(rgb_errors))
                else:
                    selected_idx = 0
                fused_pose = refined_poses[selected_idx]
                fuse_info = {
                    "cluster_size": 1.0,
                    "num_hypotheses": float(hyp_count),
                    "anchor_idx": float(selected_idx),
                    "selected_idx": float(selected_idx),
                    "used_top1_fallback": 1.0 if selected_idx == 0 else 0.0,
                }
            else:
                fused_pose, fuse_info = fuse_candidate_poses(
                    refined_poses,
                    method=pose_fusion,
                    retrieval_scores=candidate_scores,
                    consensus_radius_m=consensus_radius_m,
                    consensus_min_size=consensus_min_size,
                )
            fused_pose = fused_pose.unsqueeze(0)

            # ── Cascade refinement rounds ──
            if cascade_rounds > 0:
                for _cr in range(cascade_rounds):
                    fused_pose = cascade_refine_single(
                        model=model,
                        gaussians=gaussians,
                        dcff_renderer=dcff_renderer,
                        feat_sharp=feat_sharp,
                        query_fine=query_fine_i,
                        query_coarse=query_coarse_i,
                        best_pose=fused_pose,
                        K=K,
                        render_intr=render_intr,
                        outer_iters=outer_iters,
                        solver=solver,
                        irls_iters=irls_iters,
                        robust_kernel=robust_kernel,
                        direct_refine_iters=direct_refine_iters,
                        cascade_starts=cascade_starts,
                        cascade_noise_deg=cascade_noise_deg,
                        cascade_noise_m=cascade_noise_m,
                        render_h=render_h,
                        render_w=render_w,
                    )

            fused_render, _ = render_batch(
                gaussians, dcff_renderer, feat_sharp, fused_pose, K, render_h, render_w
            )

            oracle_idx = int(torch.argmin(final_trans_h).item())
            oracle_rot = float(final_rot_h[oracle_idx].item())
            oracle_trans = float(final_trans_h[oracle_idx].item())
            fused_rot, fused_trans = compute_pose_errors(fused_pose, pose_gt_i)
            top1_final_rot, top1_final_trans = compute_pose_errors(top1_final_pose, pose_gt_i)

            init_rot_all.append(float(init_rot_h[0].item()))
            init_trans_all.append(float(init_trans_h[0].item()))
            top1_final_rot_all.append(float(top1_final_rot[0].item()))
            top1_final_trans_all.append(float(top1_final_trans[0].item()))
            fused_final_rot_all.append(float(fused_rot[0].item()))
            fused_final_trans_all.append(float(fused_trans[0].item()))
            oracle_final_rot_all.append(oracle_rot)
            oracle_final_trans_all.append(oracle_trans)
            cluster_sizes.append(float(fuse_info["cluster_size"]))
            used_top1_fallback.append(float(fuse_info["used_top1_fallback"]))

            record = {
                "image_name": batch["image_name"][i],
                "image_stem": batch["image_stem"][i],
                "init_source": batch["init_source"][i],
                "retrieval_frame_id": int(batch["retrieval_frame_id"][i]),
                "retrieval_image_name": batch["retrieval_image_name"][i],
                "retrieval_score": float(batch["retrieval_score"][i]),
                "num_hypotheses": hyp_count,
                "fusion_method": pose_fusion,
                "cluster_size": float(fuse_info["cluster_size"]),
                "used_top1_fallback": float(fuse_info["used_top1_fallback"]),
                "anchor_idx": int(fuse_info["anchor_idx"]),
                "selected_idx": int(fuse_info["selected_idx"]),
                "init_rot_err_deg": float(init_rot_h[0].item()),
                "init_trans_err_mm": float(init_trans_h[0].item()),
                "top1_final_rot_err_deg": float(top1_final_rot[0].item()),
                "top1_final_trans_err_mm": float(top1_final_trans[0].item()),
                "fused_final_rot_err_deg": float(fused_rot[0].item()),
                "fused_final_trans_err_mm": float(fused_trans[0].item()),
                "oracle_final_rot_err_deg": oracle_rot,
                "oracle_final_trans_err_mm": oracle_trans,
                "best_candidate_rank": oracle_idx,
                "best_candidate_image_name": candidate_image_names[oracle_idx] if oracle_idx < len(candidate_image_names) else "",
                "best_candidate_score": float(candidate_scores[oracle_idx].item()),
            }
            records.append(record)

            if qual_saved < qual_limit:
                rgb_path = batch["query_rgb_path"][i]
                if rgb_path and os.path.isfile(rgb_path):
                    query_rgb = load_rgb_image(rgb_path, target_hw=(render_h, render_w))
                else:
                    query_rgb = np.zeros((render_h, render_w, 3), dtype=np.uint8)
                caption_lines = [
                    f"{record['image_name']} [{record['init_source']}] topk={hyp_count} fusion={pose_fusion}",
                    f"init(top1): {record['init_rot_err_deg']:.2f} deg / {record['init_trans_err_mm']:.1f} mm",
                    f"final(top1): {record['top1_final_rot_err_deg']:.2f} deg / {record['top1_final_trans_err_mm']:.1f} mm",
                    f"final(fused): {record['fused_final_rot_err_deg']:.2f} deg / {record['fused_final_trans_err_mm']:.1f} mm",
                    f"oracle(k): {record['oracle_final_rot_err_deg']:.2f} deg / {record['oracle_final_trans_err_mm']:.1f} mm",
                    f"cluster={int(record['cluster_size'])} fallback={int(record['used_top1_fallback'])}",
                    f"db(top1): {record['retrieval_image_name']} score={record['retrieval_score']:.4f}",
                ]
                save_triptych(
                    query_rgb,
                    feature_map_to_rgb(init_render_top1[0]),
                    feature_map_to_rgb(fused_render[0] if pose_fusion != "none" else top1_final_render[0]),
                    os.path.join(qual_dir, f"{qual_saved:04d}_{record['image_stem']}.png"),
                    caption_lines=caption_lines,
                )
                report_path = os.path.join(qual_dir, f"{qual_saved:04d}_{record['image_stem']}.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(caption_lines) + "\n")
                qual_saved += 1

    model.gru_iters = original_gru

    init_summary = summarize_pose_metrics(np.array(init_rot_all), np.array(init_trans_all))
    top1_final_summary = summarize_pose_metrics(
        np.array(top1_final_rot_all), np.array(top1_final_trans_all)
    )
    fused_final_summary = summarize_pose_metrics(
        np.array(fused_final_rot_all), np.array(fused_final_trans_all)
    )
    oracle_final_summary = summarize_pose_metrics(
        np.array(oracle_final_rot_all), np.array(oracle_final_trans_all)
    )
    return {
        "init": init_summary,
        "top1_final": top1_final_summary,
        "final": fused_final_summary,
        "oracle_final": oracle_final_summary,
        "fusion_stats": {
            "cluster_size_mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "cluster_size_median": float(np.median(cluster_sizes)) if cluster_sizes else 0.0,
            "top1_fallback_rate": float(np.mean(used_top1_fallback) * 100.0) if used_top1_fallback else 0.0,
        },
    }, records


def main():
    parser = argparse.ArgumentParser(description="Real-init evaluation for concat localizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--outer_iters", type=int, default=None)
    parser.add_argument("--gru_iters", type=int, default=None)
    parser.add_argument(
        "--solver",
        default="default",
        choices=[
            "default",
            "wls_full",
            "pnp",
            "hybrid",
            "irls2",
            "irls3_gnc",
            "direct",
            "flow+direct5",
            "flow+direct10",
            "flow+direct20",
            "direct_neg",
            "flow+direct5_neg",
            "flow+direct5_highdamp",
            "flow+direct5_neg_highdamp",
        ],
    )
    parser.add_argument("--retrieval_method", default="auto", choices=["auto", "cls", "nearest_train_pose_gt"])
    parser.add_argument("--retrieval_feature_dir", default=None)
    parser.add_argument("--init_poses_path", default=None)
    parser.add_argument("--save_init_poses_path", default=None)
    parser.add_argument(
        "--fallback_mode",
        default="nearest_train_pose_gt",
        choices=["nearest_train_pose_gt", "synthetic_noise"],
    )
    parser.add_argument("--fallback_noise_deg", type=float, default=3.0)
    parser.add_argument("--fallback_noise_m", type=float, default=0.10)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--qual_limit", type=int, default=12)
    parser.add_argument("--qual_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--retrieval_topk", type=int, default=1)
    parser.add_argument(
        "--pose_fusion",
        default="none",
        choices=["none", "centroid", "consensus", "consensus_centroid", "rgb_select"],
        help="How to combine refined top-k hypotheses.",
    )
    parser.add_argument("--consensus_radius_m", type=float, default=1.0)
    parser.add_argument("--consensus_min_size", type=int, default=2)
    parser.add_argument("--cascade_rounds", type=int, default=0,
                        help="Number of cascade refinement rounds after initial refinement")
    parser.add_argument("--cascade_starts", type=int, default=10,
                        help="Number of multi-start perturbations per cascade round")
    parser.add_argument("--cascade_noise_deg", type=float, default=0.3,
                        help="Rotation noise std for cascade perturbations")
    parser.add_argument("--cascade_noise_m", type=float, default=0.05,
                        help="Translation noise std for cascade perturbations")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    config = load_mainline_config(args.config)

    exp_name = config.get("exp_name", Path(args.checkpoint).stem)
    out_root = args.output_dir or os.path.join(config.get("output_dir", "output"), exp_name)
    run_tag = "real_init"
    if args.retrieval_topk > 1 or args.pose_fusion != "none":
        run_tag = f"real_init_top{args.retrieval_topk}_{args.pose_fusion}"
    if args.cascade_rounds > 0:
        run_tag += f"_cascade{args.cascade_rounds}x{args.cascade_starts}"
    qual_dir = args.qual_dir or os.path.join(out_root, f"qual_{run_tag}")
    real_init_dir = os.path.join(out_root, run_tag)
    os.makedirs(qual_dir, exist_ok=True)
    os.makedirs(real_init_dir, exist_ok=True)

    retrieval_feature_dir = infer_retrieval_feature_dir(config, args.retrieval_feature_dir)
    default_init_name = (
        f"retrieval_init_poses_top{args.retrieval_topk}.npz"
        if args.retrieval_topk > 1
        else "retrieval_init_poses.npz"
    )
    init_poses_path = args.init_poses_path or os.path.join(real_init_dir, default_init_name)
    save_init_poses_path = args.save_init_poses_path or init_poses_path

    print("Building DCFF rendering pipeline...")
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    pose_ckpt = load_concat_pose_checkpoint(args.checkpoint, device)
    restored_map = apply_localization_map_state(
        dcff_renderer,
        feat_sharp,
        pose_ckpt,
        printer=print,
    )
    if restored_map:
        print(f"Restored localization map state: {', '.join(restored_map)}")

    ds_cfg = config["dataset"]
    dcff_cfg = config["dcff"]
    render_h = dcff_cfg.get("render_height", 68)
    render_w = dcff_cfg.get("render_width", 120)
    fine_hw = tuple(ds_cfg.get("fine_hw", [render_h, render_w]))
    coarse_hw = tuple(ds_cfg.get("coarse_hw", fine_hw))
    use_coarse = config.get("model", {}).get("use_coarse", True)

    colmap_cameras = read_colmap_cameras(os.path.join(ds_cfg["colmap_dir"], "cameras.bin"))
    first_cam = next(iter(colmap_cameras.values()))
    model.BASE_INTRINSICS = camera_params_to_intrinsics(first_cam)
    model.IMG_HW = (int(first_cam.height), int(first_cam.width))

    val_ds = RadioLocRetrievalDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        split="test",
        split_file=ds_cfg["test_split"],
        retrieval_train_split=ds_cfg["train_split"],
        retrieval_feature_dir=retrieval_feature_dir,
        retrieval_method=args.retrieval_method,
        retrieval_topk=args.retrieval_topk,
        source_dir=ds_cfg.get("source_dir"),
        init_poses_path=init_poses_path if os.path.isfile(init_poses_path) else None,
        save_init_poses_path=save_init_poses_path,
        fallback_mode=args.fallback_mode,
        fallback_noise_rot_deg=args.fallback_noise_deg,
        fallback_noise_trans_m=args.fallback_noise_m,
        fine_hw=fine_hw,
        coarse_hw=coarse_hw,
        cache_in_memory=True,
        noise_rot_deg=args.fallback_noise_deg,
        noise_trans_m=args.fallback_noise_m,
        normalize_features=ds_cfg.get("normalize_features", False),
    )

    eval_dataset = val_ds
    if args.max_samples > 0:
        eval_dataset = Subset(val_ds, list(range(min(args.max_samples, len(val_ds)))))

    val_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    solver_configs = {
        "default": ("wls", 0, "huber", 0),
        "wls_full": ("wls_full", 0, "huber", 0),
        "pnp": ("pnp", 0, "huber", 0),
        "hybrid": ("hybrid", 0, "huber", 0),
        "irls2": ("wls", 2, "huber", 0),
        "irls3_gnc": ("wls", 3, "gnc_gm", 0),
        "direct": ("direct", 0, "huber", 0),
        "direct_neg": ("direct_neg", 0, "huber", 0),
        "flow+direct5": ("wls", 0, "huber", 5),
        "flow+direct10": ("wls", 0, "huber", 10),
        "flow+direct20": ("wls", 0, "huber", 20),
        "flow+direct5_neg": ("flow_neg", 0, "huber", 5),
        "flow+direct5_highdamp": ("flow_highdamp", 0, "huber", 5),
        "flow+direct5_neg_highdamp": ("flow_neg_highdamp", 0, "huber", 5),
    }
    solver_type, irls_iters, robust_kernel, direct_refine_iters = solver_configs[args.solver]
    gru_iters = args.gru_iters if args.gru_iters is not None else config.get("model", {}).get("gru_iters", 6)
    outer_iters = (
        args.outer_iters
        if args.outer_iters is not None
        else config.get("training", {}).get("outer_iters_val", config.get("training", {}).get("outer_iters_train", 1))
    )

    print("=" * 72)
    print(f"Real-init eval: {exp_name} epoch={ckpt_epoch}")
    print(f"Retrieval feature dir: {retrieval_feature_dir}")
    print(f"Init pose cache: {save_init_poses_path}")
    print(f"Init stats: {val_ds.init_stats}")
    print(f"Retrieval top-k: {args.retrieval_topk}  fusion: {args.pose_fusion}")
    if args.cascade_rounds > 0:
        print(f"Cascade: {args.cascade_rounds} rounds × {args.cascade_starts} starts "
              f"(noise: {args.cascade_noise_deg}°/{args.cascade_noise_m}m)")
    print("=" * 72)

    summaries, records = evaluate_real_init(
        model=model,
        gaussians=gaussians,
        dcff_renderer=dcff_renderer,
        feat_sharp=feat_sharp,
        val_loader=val_loader,
        device=device,
        outer_iters=outer_iters,
        gru_iters=gru_iters,
        render_h=render_h,
        render_w=render_w,
        use_coarse=use_coarse,
        solver=solver_type,
        irls_iters=irls_iters,
        robust_kernel=robust_kernel,
        direct_refine_iters=direct_refine_iters,
        qual_dir=qual_dir,
        qual_limit=args.qual_limit,
        retrieval_topk=args.retrieval_topk,
        pose_fusion=args.pose_fusion,
        consensus_radius_m=args.consensus_radius_m,
        consensus_min_size=args.consensus_min_size,
        cascade_rounds=args.cascade_rounds,
        cascade_starts=args.cascade_starts,
        cascade_noise_deg=args.cascade_noise_deg,
        cascade_noise_m=args.cascade_noise_m,
    )

    source_counts = Counter(r["init_source"] for r in records)
    readiness = "ready" if any(not src.startswith("fallback_") for src in source_counts) else "blocked"
    blocker = ""
    if readiness != "ready":
        blocker = "No non-fallback init was produced; all samples fell back."

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "output_dir": out_root,
        "epoch": ckpt_epoch,
        "solver": args.solver,
        "outer_iters": outer_iters,
        "gru_iters": gru_iters,
        "retrieval_topk": args.retrieval_topk,
        "pose_fusion": args.pose_fusion,
        "consensus_radius_m": args.consensus_radius_m,
        "num_samples": len(records),
        "retrieval_feature_dir": retrieval_feature_dir,
        "init_pose_cache": save_init_poses_path,
        "dataset_init_stats": val_ds.init_stats,
        "counts_by_source": dict(source_counts),
        "init_metrics": summaries["init"],
        "top1_final_metrics": summaries["top1_final"],
        "final_metrics": summaries["final"],
        "full_pipeline_metrics": summarize_full_pipeline_metrics(
            init_rot_errs=[r["init_rot_err_deg"] for r in records],
            init_trans_errs=[r["init_trans_err_mm"] for r in records],
            final_rot_errs=[r["fused_final_rot_err_deg"] for r in records],
            final_trans_errs=[r["fused_final_trans_err_mm"] for r in records],
        ),
        "oracle_final_metrics": summaries["oracle_final"],
        "fusion_stats": summaries["fusion_stats"],
        "readiness": readiness,
        "blocker": blocker,
        "qual_dir": qual_dir,
    }

    summary_json = os.path.join(real_init_dir, "summary.json")
    summary_txt = os.path.join(real_init_dir, "summary.txt")
    metrics_csv = os.path.join(real_init_dir, "sample_metrics.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"readiness: {readiness}\n")
        if blocker:
            f.write(f"blocker: {blocker}\n")
        f.write(f"counts_by_source: {dict(source_counts)}\n")
        f.write(f"init_metrics: {json.dumps(summaries['init'])}\n")
        f.write(f"top1_final_metrics: {json.dumps(summaries['top1_final'])}\n")
        f.write(f"final_metrics: {json.dumps(summaries['final'])}\n")
        f.write(f"oracle_final_metrics: {json.dumps(summaries['oracle_final'])}\n")
        f.write(f"fusion_stats: {json.dumps(summaries['fusion_stats'])}\n")
        f.write(f"qual_dir: {qual_dir}\n")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["image_name"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    save_experiment_bundle(
        exp_name=run_tag,
        output_dir=real_init_dir,
        metrics=summary,
        summary_lines=[
            f"init median: {summaries['init']['rot_median']:.3f} deg / {summaries['init']['trans_median']:.1f} mm",
            f"top1 final median: {summaries['top1_final']['rot_median']:.3f} deg / {summaries['top1_final']['trans_median']:.1f} mm",
            f"final median: {summaries['final']['rot_median']:.3f} deg / {summaries['final']['trans_median']:.1f} mm",
            f"oracle final median: {summaries['oracle_final']['rot_median']:.3f} deg / {summaries['oracle_final']['trans_median']:.1f} mm",
        ],
        notes=[
            f"readiness={readiness}",
            blocker if blocker else "real-init evaluation completed without blocker",
        ],
        artifact_paths=[summary_json, summary_txt, metrics_csv, qual_dir],
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
