#!/usr/bin/env python3
"""Evaluate the concat localizer under retrieval-based real pose init."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
from data.radio_loc_retrieval_dataset import (  # noqa: E402
    OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS,
    RadioLocRetrievalDataset,
)
from pose_refine import (  # noqa: E402
    load_concat_pose_checkpoint,
    load_concat_pose_model as load_model,
    load_external_local_corr_projector,
    run_model_refine_iteration,
)
from pose_refine.runtime import apply_pose_delta  # noqa: E402
from pose_refine.utils.geometry_solver import feature_metric_solve, pnp_ransac_solve  # noqa: E402
from feature_field import build_dcff, intrinsics_to_K, render_batch, render_feature_bundle_batch  # noqa: E402
from feature_field.runtime import apply_localization_map_state  # noqa: E402
from feature_field.utils.loc_reporting import save_experiment_bundle  # noqa: E402
from feature_field.utils.project_config import (  # noqa: E402
    load_mainline_config,
    should_restore_pose_checkpoint_map_state,
)
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


def sha256_file_or_none(path: str | os.PathLike | None) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def summarize_init_candidate_slots(dataset) -> Dict[str, int]:
    entries = list(getattr(dataset, "init_by_name", {}).values())
    if not entries:
        return {
            "loaded_candidate_slots": 0,
            "loaded_valid_candidates_min": 0,
            "loaded_valid_candidates_max": 0,
        }
    slots = []
    valid_counts = []
    for entry in entries:
        candidates = np.asarray(entry.get("pose_init_candidates", []))
        valid = np.asarray(entry.get("candidate_valid_mask", []), dtype=bool)
        slots.append(int(candidates.shape[0]) if candidates.ndim >= 1 else 0)
        valid_counts.append(int(valid.sum()) if valid.size else 0)
    return {
        "loaded_candidate_slots": int(max(slots) if slots else 0),
        "loaded_valid_candidates_min": int(min(valid_counts) if valid_counts else 0),
        "loaded_valid_candidates_max": int(max(valid_counts) if valid_counts else 0),
    }


def normalize_candidate_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    finite = torch.isfinite(scores)
    if not bool(finite.any()):
        return torch.zeros_like(scores)
    clean = torch.where(finite, scores, scores[finite].min())
    spread = clean.max() - clean.min()
    if float(spread.item()) <= 1e-8:
        return torch.zeros_like(clean)
    return (clean - clean.min()) / (spread + 1e-8)


def compute_pnp_quality_prior(
    *,
    fallback_scores: torch.Tensor,
    candidate_quality: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build a per-query PnP quality prior from saved candidate fields."""

    def _field(name: str, default: float) -> torch.Tensor:
        value = candidate_quality.get(name)
        if value is None:
            return torch.full_like(fallback_scores.float(), float(default))
        return value.to(device=fallback_scores.device, dtype=torch.float32)

    success = _field("retrieval_pnp_success_candidates", 1.0).clamp(0.0, 1.0)
    inliers = _field("retrieval_pnp_num_inliers_candidates", 0.0).clamp(min=0.0)
    ratio = _field("retrieval_pnp_inlier_ratio_candidates", 0.0).clamp(min=0.0)
    conf = _field("retrieval_pnp_inlier_conf_mean_candidates", 0.0).clamp(min=0.0)
    rmse = _field("retrieval_pnp_reproj_rmse_candidates", float("inf")).clamp(min=0.0)
    original = _field("retrieval_original_scores_candidates", 0.0)
    rmse = torch.where(torch.isfinite(rmse), rmse, torch.full_like(rmse, 1e6))

    quality = (
        normalize_candidate_scores(torch.log1p(inliers))
        + normalize_candidate_scores(ratio)
        + 0.5 * normalize_candidate_scores(conf)
        - normalize_candidate_scores(torch.log1p(rmse))
        + 0.25 * normalize_candidate_scores(original)
    )
    if float(success.sum().item()) > 0.0:
        quality = quality.masked_fill(success <= 0.0, float("-inf"))
    if bool(torch.isfinite(quality).any()):
        return quality
    return fallback_scores.float()


def compute_candidate_support_counts(poses_w2c: torch.Tensor, radius_m: float) -> torch.Tensor:
    if poses_w2c.numel() == 0:
        return torch.empty((0,), device=poses_w2c.device, dtype=torch.float32)
    centers = camera_centers_from_w2c(poses_w2c)
    pairwise = torch.cdist(centers, centers)
    return (pairwise <= float(radius_m)).sum(dim=1).float()


def compute_linear_quality_weights(scores: torch.Tensor, min_weight: float = 0.1) -> torch.Tensor:
    weights = normalize_candidate_scores(scores.float()) + float(min_weight)
    return torch.clamp(weights, min=float(min_weight))


def eval_apply_pose_delta(model, pose_w2c: torch.Tensor, delta_xi: torch.Tensor, outer_iter: int) -> torch.Tensor:
    update_scale = float(getattr(model, "pose_update_scale", 1.0))
    trans_scale = float(getattr(model, "pose_update_trans_scale", 1.0))
    rot_scale = float(getattr(model, "pose_update_rot_scale", 1.0))
    if int(outer_iter) > 0:
        trans_after = getattr(model, "pose_update_trans_scale_after_first", None)
        rot_after = getattr(model, "pose_update_rot_scale_after_first", None)
        if trans_after is not None:
            trans_scale = float(trans_after)
        if rot_after is not None:
            rot_scale = float(rot_after)
    return apply_pose_delta(
        pose_w2c.float(),
        delta_xi.float(),
        scale=update_scale,
        trans_scale=trans_scale,
        rot_scale=rot_scale,
    )


def build_refine_gate_config(args) -> Dict[str, float]:
    gate = {}
    for cli_name, gate_name in [
        ("refine_gate_max_step_trans_mm", "max_step_trans_mm"),
        ("refine_gate_max_step_rot_deg", "max_step_rot_deg"),
        ("refine_gate_max_delta_xi_norm", "max_delta_xi_norm"),
        ("refine_gate_max_flow_mag_mean", "max_flow_mag_mean"),
        ("refine_gate_max_flow_mag_max", "max_flow_mag_max"),
        ("refine_gate_min_confidence_mean", "min_confidence_mean"),
        ("refine_gate_max_confidence_lowfrac", "max_confidence_lowfrac"),
        ("refine_gate_min_depth_valid_ratio", "min_depth_valid_ratio"),
    ]:
        value = getattr(args, cli_name, None)
        if value is not None and float(value) > 0.0:
            gate[gate_name] = float(value)
    return gate


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
    quality_scores: Optional[torch.Tensor] = None,
    quality_consensus_large_size: int = 15,
    quality_consensus_small_size: int = 10,
    quality_consensus_quality_weight: float = 0.7,
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
    if method == "score_select":
        selected_idx = 0
        if retrieval_scores is not None and retrieval_scores.numel() == num_hyp:
            selected_idx = int(torch.argmax(retrieval_scores.float()).item())
        return poses_w2c[selected_idx], {
            "cluster_size": 1.0,
            "num_hypotheses": float(num_hyp),
            "anchor_idx": float(selected_idx),
            "selected_idx": float(selected_idx),
            "used_top1_fallback": 1.0 if selected_idx == 0 else 0.0,
        }

    centers = camera_centers_from_w2c(poses_w2c)
    pairwise = torch.cdist(centers, centers)
    support = (pairwise <= float(consensus_radius_m)).sum(dim=1).float()

    if method == "quality_consensus":
        scores_for_quality = quality_scores if quality_scores is not None else retrieval_scores
        if scores_for_quality is not None and scores_for_quality.numel() == num_hyp:
            quality_norm = normalize_candidate_scores(scores_for_quality.to(poses_w2c.device).float())
        else:
            quality_norm = torch.zeros_like(support)
        support_norm = support / max(1.0, float(num_hyp))
        joint = support_norm + float(quality_consensus_quality_weight) * quality_norm
        anchor_idx = int(torch.argmax(joint).item())
        cluster_mask = pairwise[anchor_idx] <= float(consensus_radius_m)
        cluster_size = int(cluster_mask.sum().item())
        if cluster_size >= int(quality_consensus_large_size):
            cluster_quality = torch.exp(quality_norm[cluster_mask] - pairwise[anchor_idx][cluster_mask])
            fused = fuse_pose_centroid(poses_w2c[cluster_mask], weights=cluster_quality)
            selected_idx = anchor_idx
        elif cluster_size >= int(quality_consensus_small_size):
            fused = poses_w2c[anchor_idx]
            selected_idx = anchor_idx
        else:
            selected_idx = int(torch.argmax(quality_norm).item()) if num_hyp > 1 else 0
            fused = poses_w2c[selected_idx]
        return fused, {
            "cluster_size": float(cluster_size),
            "num_hypotheses": float(num_hyp),
            "anchor_idx": float(anchor_idx),
            "selected_idx": float(selected_idx),
            "used_top1_fallback": 1.0 if selected_idx == 0 else 0.0,
        }

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

    if method == "quality_weighted_consensus_centroid":
        scores_for_quality = quality_scores if quality_scores is not None else retrieval_scores
        if scores_for_quality is not None and scores_for_quality.numel() == num_hyp:
            weights = compute_linear_quality_weights(scores_for_quality.to(poses_w2c.device).float())
            fused_pose = fuse_pose_centroid(poses_w2c[cluster_mask], weights=weights[cluster_mask])
        else:
            fused_pose = fuse_pose_centroid(poses_w2c[cluster_mask])
        selected_idx = anchor_idx
    elif method == "consensus_centroid":
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


def compute_candidate_feature_residuals(
    query_fine: torch.Tensor,
    rendered_fine: torch.Tensor,
    *,
    model=None,
    normalize: bool = True,
) -> torch.Tensor:
    """Score candidate poses by query-map feature disagreement."""
    if query_fine.shape[-2:] != rendered_fine.shape[-2:]:
        query_fine = F.interpolate(
            query_fine,
            size=rendered_fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if query_fine.shape[0] == 1 and rendered_fine.shape[0] > 1:
        query_fine = query_fine.expand(rendered_fine.shape[0], -1, -1, -1)
    if (
        model is not None
        and bool(getattr(model, "feature_select_use_local_corr_projection", True))
        and getattr(model, "external_local_corr_projector", None) is not None
        and hasattr(model, "_project_for_local_corr")
    ):
        q, r = model._project_for_local_corr(query_fine.float(), rendered_fine.float())
        if q.shape[0] == 1 and r.shape[0] > 1:
            q = q.expand(r.shape[0], -1, -1, -1)
        return (q.float() - r.float()).pow(2).mean(dim=(1, 2, 3))
    q = query_fine.float()
    r = rendered_fine.float()
    if normalize:
        q = F.normalize(q, dim=1)
        r = F.normalize(r, dim=1)
    return (q - r).pow(2).mean(dim=(1, 2, 3))


def project_featuremetric_pair(model, query_fine: torch.Tensor, rendered_fine: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if query_fine.shape[-2:] != rendered_fine.shape[-2:]:
        query_fine = F.interpolate(
            query_fine,
            size=rendered_fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if (
        bool(getattr(model, "direct_featuremetric_use_local_corr_projection", False))
        and getattr(model, "external_local_corr_projector", None) is not None
        and hasattr(model, "_project_for_local_corr")
    ):
        return model._project_for_local_corr(query_fine.float(), rendered_fine.float())
    return F.normalize(query_fine.float(), dim=1), F.normalize(rendered_fine.float(), dim=1)


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
    refine_gate: Optional[Dict[str, float]] = None,
):
    pose_cur = pose_init.clone()
    use_flow = not solver.startswith("direct")
    n_flow_iters = outer_iters if use_flow else 0
    batch_size = pose_init.shape[0]
    gate_cfg = dict(refine_gate or {})
    need_diagnostics = collect_diagnostics or bool(gate_cfg)
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
        "delta_trans_m": [],
        "delta_rot_deg": [],
        "step_trans_mm": [],
        "step_rot_deg": [],
        "depth_valid_ratio": [],
        "gate_keep": [],
        "gate_rejected": [],
    }
    last_depth = None

    def spatial_depth_mask(depth_tensor: torch.Tensor) -> torch.Tensor:
        mask = depth_tensor > 0.05
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        return mask.bool()

    def compute_update_diagnostics(pred: Dict, depth: torch.Tensor, delta_xi: torch.Tensor, proposed_pose: torch.Tensor, base_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        depth_mask = spatial_depth_mask(depth)
        flow_map = pred.get("flow")
        if flow_map is not None:
            flow_map = flow_map.float()
            flow_mag = torch.norm(flow_map, dim=1)
            flow_mean, flow_std = masked_mean_std_2d(flow_mag, depth_mask)
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
        else:
            flow_mean = zero_stat
            flow_std = zero_stat
            flow_max = zero_stat
            flow_step_mean = zero_stat

        conf_map = pred.get("confidence")
        if conf_map is not None:
            conf_map = conf_map.float().mean(dim=1)
            conf_mean, conf_std = masked_mean_std_2d(conf_map, depth_mask)
            denom = depth_mask.float().sum(dim=(-1, -2)).clamp(min=1.0)
            conf_lowfrac = ((conf_map < 0.25) & depth_mask).float().sum(dim=(-1, -2)) / denom
        else:
            conf_mean = zero_stat
            conf_std = zero_stat
            conf_lowfrac = torch.ones_like(zero_stat)

        step_rot, step_trans = compute_pose_errors(proposed_pose, base_pose)
        depth_valid_ratio = depth_mask.float().mean(dim=(-1, -2))
        delta = delta_xi.float()
        return {
            "flow_mag_mean": flow_mean,
            "flow_mag_std": flow_std,
            "flow_mag_max": flow_max,
            "confidence_mean": conf_mean,
            "confidence_std": conf_std,
            "confidence_lowfrac": conf_lowfrac,
            "flow_step_mean": flow_step_mean,
            "delta_xi_norm": delta.norm(dim=1),
            "delta_trans_m": delta[:, :3].norm(dim=1),
            "delta_rot_deg": delta[:, 3:].norm(dim=1) * (180.0 / math.pi),
            "step_trans_mm": step_trans.float(),
            "step_rot_deg": step_rot.float(),
            "depth_valid_ratio": depth_valid_ratio,
        }

    def compute_gate_keep(stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        keep = torch.ones(batch_size, device=pose_init.device, dtype=torch.bool)
        for key, threshold in [
            ("step_trans_mm", gate_cfg.get("max_step_trans_mm")),
            ("step_rot_deg", gate_cfg.get("max_step_rot_deg")),
            ("delta_xi_norm", gate_cfg.get("max_delta_xi_norm")),
            ("flow_mag_mean", gate_cfg.get("max_flow_mag_mean")),
            ("flow_mag_max", gate_cfg.get("max_flow_mag_max")),
            ("confidence_lowfrac", gate_cfg.get("max_confidence_lowfrac")),
        ]:
            if threshold is not None:
                keep = keep & (stats[key] <= float(threshold))
        for key, threshold in [
            ("confidence_mean", gate_cfg.get("min_confidence_mean")),
            ("depth_valid_ratio", gate_cfg.get("min_depth_valid_ratio")),
        ]:
            if threshold is not None:
                keep = keep & (stats[key] >= float(threshold))
        return keep

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
                delta_xi = pnp_xi.float()
        elif solver == "wls_full" and "delta_xi_full" in pred:
            with torch.cuda.amp.autocast(enabled=False):
                delta_xi = pred["delta_xi_full"].float()
        elif "delta_xi" in pred:
            with torch.cuda.amp.autocast(enabled=False):
                delta_xi = pred["delta_xi"].float()
        else:
            continue

        with torch.cuda.amp.autocast(enabled=False):
            proposed_pose = eval_apply_pose_delta(model, pose_mid, delta_xi, outer_i)

        if need_diagnostics:
            stats = compute_update_diagnostics(pred, depth, delta_xi, proposed_pose, pose_mid)
            gate_keep = compute_gate_keep(stats) if gate_cfg else torch.ones_like(zero_stat, dtype=torch.bool)
            if gate_cfg:
                pose_cur = torch.where(gate_keep[:, None, None], proposed_pose, pose_mid)
            else:
                pose_cur = proposed_pose

            for key, value in stats.items():
                diag_acc[key].append(value)
            diag_acc["gate_keep"].append(gate_keep.float())
            diag_acc["gate_rejected"].append((~gate_keep).float())
        else:
            pose_cur = proposed_pose

    n_direct = direct_refine_iters if solver != "direct" else outer_iters
    for _ in range(n_direct):
        ref_fine, depth = render_batch(gaussians, dcff_renderer, feat_sharp, pose_cur, K, *query_fine.shape[-2:])
        last_depth = depth
        with torch.cuda.amp.autocast(enabled=False):
            depth_s = depth.float() if depth.dim() == 3 else depth.squeeze(1).float()
            damping = 1.0 if "highdamp" in solver else 1e-2
            fm_query_fine, fm_ref_fine = project_featuremetric_pair(model, query_fine, ref_fine)
            direct_xi, _ = feature_metric_solve(
                fm_query_fine.float(), fm_ref_fine.float(), depth_s, render_intr, damping=damping
            )
            if "neg" in solver:
                direct_xi = -direct_xi
            pose_cur = eval_apply_pose_delta(model, pose_cur, direct_xi, n_flow_iters)

    final_render, final_depth = render_batch(gaussians, dcff_renderer, feat_sharp, pose_cur, K, *query_fine.shape[-2:])
    if last_depth is None:
        last_depth = final_depth
    diagnostics = {}
    for key, values in diag_acc.items():
        diagnostics[key] = torch.stack(values, dim=0).mean(dim=0) if values else zero_stat
    if not diag_acc["depth_valid_ratio"]:
        diagnostics["depth_valid_ratio"] = spatial_depth_mask(last_depth).float().mean(dim=(-1, -2))
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


def choose_loftr_validation_indices(
    *,
    candidate_scores: torch.Tensor,
    candidate_quality_scores: torch.Tensor,
    candidate_support_counts: torch.Tensor,
    topn: int,
) -> List[int]:
    num_hyp = int(candidate_scores.numel())
    if num_hyp <= 0:
        return []
    if int(topn) <= 0 or int(topn) >= num_hyp:
        return list(range(num_hyp))

    support_norm = normalize_candidate_scores(candidate_support_counts.float())
    quality_norm = normalize_candidate_scores(candidate_quality_scores.float())
    score_norm = normalize_candidate_scores(candidate_scores.float())
    joint = support_norm + 0.3 * quality_norm + 1e-3 * score_norm
    seeds = [
        0,
        int(torch.argmax(candidate_support_counts.float()).item()),
        int(torch.argmax(candidate_quality_scores.float()).item()),
    ]
    ordered = torch.argsort(joint, descending=True).detach().cpu().tolist()
    indices: List[int] = []
    for idx in seeds + [int(v) for v in ordered]:
        if 0 <= idx < num_hyp and idx not in indices:
            indices.append(idx)
        if len(indices) >= int(topn):
            break
    return indices


def camera_center_from_w2c_np(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    return (-pose[:3, :3].T @ pose[:3, 3]).astype(np.float64)


def compute_loftr_validation_score(
    result,
    validated_pose_w2c: np.ndarray,
    refined_pose_w2c: np.ndarray,
    *,
    score_mode: str = "basic",
) -> Dict[str, float]:
    inliers = float(getattr(result, "num_inliers", 0))
    matches = float(getattr(result, "num_confident_matches", 0))
    raw_matches = float(getattr(result, "num_raw_matches", 0))
    depth_valid = float(getattr(result, "num_depth_valid", 0))
    mean_conf = float(getattr(result, "mean_confidence", 0.0))
    delta_m = float(
        np.linalg.norm(
            camera_center_from_w2c_np(validated_pose_w2c)
            - camera_center_from_w2c_np(refined_pose_w2c)
        )
    )
    quality = (getattr(result, "extra", {}) or {}).get("pnp_quality", {}) or {}
    reproj_rmse = float(quality.get("pnp_reproj_rmse", np.inf))
    reproj_median = float(quality.get("pnp_reproj_median", np.inf))
    inlier_ratio = float(quality.get("pnp_inlier_ratio", inliers / max(matches, 1.0)))
    inlier_conf = float(quality.get("pnp_inlier_conf_mean", mean_conf))
    if score_mode == "pnp_quality":
        base_score = (
            inliers
            * max(inlier_ratio, 1e-6)
            * max(inlier_conf, 1e-6)
            / (1.0 + max(reproj_rmse, 0.0))
        )
    else:
        base_score = inliers * max(mean_conf, 1e-6)
    return {
        "score": float(base_score / (1.0 + max(delta_m, 0.0))),
        "inliers": inliers,
        "matches": matches,
        "raw_matches": raw_matches,
        "depth_valid": depth_valid,
        "mean_conf": mean_conf,
        "consistency_m": delta_m,
        "reproj_rmse": reproj_rmse,
        "reproj_median": reproj_median,
        "inlier_ratio": inlier_ratio,
        "inlier_conf_mean": inlier_conf,
    }


@torch.no_grad()
def run_loftr_render_validation(
    *,
    loftr_verifier,
    gaussians,
    query_rgb: np.ndarray,
    refined_poses: torch.Tensor,
    candidate_indices: List[int],
    intrinsics: Dict[str, float],
    orig_hw: Tuple[int, int],
    loftr_hw: Tuple[int, int],
    device: torch.device,
    use_validated_pose: bool,
    score_mode: str = "basic",
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, List[float]]]:
    from pose_refine.evaluate_pipeline import _render_rgbd_for_loftr
    from feature_retrieval.render_loftr_pnp_init_export import attach_pnp_quality_stats

    num_hyp = int(refined_poses.shape[0])
    scores = np.full((num_hyp,), -np.inf, dtype=np.float32)
    inliers = np.zeros((num_hyp,), dtype=np.float32)
    matches = np.zeros((num_hyp,), dtype=np.float32)
    raw_matches = np.zeros((num_hyp,), dtype=np.float32)
    depth_valid = np.zeros((num_hyp,), dtype=np.float32)
    mean_conf = np.zeros((num_hyp,), dtype=np.float32)
    consistency_m = np.full((num_hyp,), np.inf, dtype=np.float32)
    reproj_rmse = np.full((num_hyp,), np.inf, dtype=np.float32)
    reproj_median = np.full((num_hyp,), np.inf, dtype=np.float32)
    inlier_ratio = np.zeros((num_hyp,), dtype=np.float32)
    inlier_conf_mean = np.zeros((num_hyp,), dtype=np.float32)
    validated_poses: Dict[int, np.ndarray] = {}
    sx = float(loftr_hw[1]) / float(orig_hw[1])
    sy = float(loftr_hw[0]) / float(orig_hw[0])
    intrinsics_loftr = {
        "fx": float(intrinsics["fx"]) * sx,
        "fy": float(intrinsics["fy"]) * sy,
        "cx": float(intrinsics["cx"]) * sx,
        "cy": float(intrinsics["cy"]) * sy,
    }

    for idx in candidate_indices:
        pose_np = refined_poses[idx].detach().cpu().float().numpy()
        try:
            rendered_rgb, rendered_depth = _render_rgbd_for_loftr(
                gaussians,
                pose_np,
                intrinsics,
                loftr_hw,
                orig_hw,
                device,
            )
            result = loftr_verifier.estimate_pose(
                query_rgb,
                rendered_rgb,
                rendered_depth,
                pose_np,
                intrinsics,
                orig_hw,
            )
        except Exception:
            continue
        inliers[idx] = float(getattr(result, "num_inliers", 0))
        matches[idx] = float(getattr(result, "num_confident_matches", 0))
        raw_matches[idx] = float(getattr(result, "num_raw_matches", 0))
        depth_valid[idx] = float(getattr(result, "num_depth_valid", 0))
        mean_conf[idx] = float(getattr(result, "mean_confidence", 0.0))
        if bool(getattr(result, "success", False)) and getattr(result, "pose_w2c", None) is not None:
            pose_valid = np.asarray(result.pose_w2c, dtype=np.float32)
            attach_pnp_quality_stats(result, pose_valid, intrinsics_loftr)
            validated_poses[idx] = pose_valid
            stats = compute_loftr_validation_score(
                result,
                pose_valid,
                pose_np,
                score_mode=score_mode,
            )
            scores[idx] = float(stats["score"])
            consistency_m[idx] = float(stats["consistency_m"])
            reproj_rmse[idx] = float(stats["reproj_rmse"])
            reproj_median[idx] = float(stats["reproj_median"])
            inlier_ratio[idx] = float(stats["inlier_ratio"])
            inlier_conf_mean[idx] = float(stats["inlier_conf_mean"])

    finite = np.isfinite(scores)
    if finite.any():
        selected_idx = int(np.argmax(scores))
    else:
        selected_idx = int(candidate_indices[0]) if candidate_indices else 0

    if use_validated_pose and selected_idx in validated_poses:
        fused_pose = torch.from_numpy(validated_poses[selected_idx]).to(
            device=refined_poses.device,
            dtype=refined_poses.dtype,
        )
    else:
        fused_pose = refined_poses[selected_idx]

    info = {
        "cluster_size": float(len(candidate_indices)),
        "num_hypotheses": float(num_hyp),
        "anchor_idx": float(selected_idx),
        "selected_idx": float(selected_idx),
        "used_top1_fallback": 1.0 if selected_idx == 0 else 0.0,
    }
    diagnostics = {
        "scores": scores.astype(float).tolist(),
        "inliers": inliers.astype(float).tolist(),
        "matches": matches.astype(float).tolist(),
        "raw_matches": raw_matches.astype(float).tolist(),
        "depth_valid": depth_valid.astype(float).tolist(),
        "mean_conf": mean_conf.astype(float).tolist(),
        "consistency_m": consistency_m.astype(float).tolist(),
        "reproj_rmse": reproj_rmse.astype(float).tolist(),
        "reproj_median": reproj_median.astype(float).tolist(),
        "inlier_ratio": inlier_ratio.astype(float).tolist(),
        "inlier_conf_mean": inlier_conf_mean.astype(float).tolist(),
    }
    return fused_pose, info, diagnostics


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
    quality_consensus_large_size: int = 15,
    quality_consensus_small_size: int = 10,
    quality_consensus_quality_weight: float = 0.7,
    hyp_chunk_size: int = 0,
    cascade_rounds: int = 0,
    cascade_starts: int = 10,
    cascade_noise_deg: float = 0.3,
    cascade_noise_m: float = 0.05,
    record_refine_diagnostics: bool = False,
    refine_gate: Optional[Dict[str, float]] = None,
    loftr_select_topn: int = 0,
    loftr_select_long_edge: int = 640,
    loftr_select_conf: float = 0.3,
    loftr_select_min_matches: int = 6,
    loftr_select_reproj_threshold: float = 8.0,
    loftr_select_pnp_iters: int = 2000,
    loftr_select_use_magsac: bool = False,
    loftr_select_score_mode: str = "basic",
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
    diag_record_values: Dict[str, List[float]] = {}
    records: List[Dict] = []
    qual_saved = 0
    gate_cfg = dict(refine_gate or {})
    collect_refine_diagnostics = record_refine_diagnostics or bool(gate_cfg)
    loftr_verifier = None

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
                candidate_limit = min(int(retrieval_topk), int(pose_init_candidates.shape[1]))
                valid_mask = candidate_valid_mask[i][:candidate_limit]
                candidate_poses = pose_init_candidates[i][:candidate_limit][valid_mask].to(device)
                candidate_scores = retrieval_scores_candidates[i][:candidate_limit][valid_mask].to(device)
                candidate_frame_ids = retrieval_frame_ids_candidates[i][:candidate_limit][valid_mask].cpu().tolist()
                candidate_image_names = [
                    retrieval_image_names_candidates[i][j]
                    for j, keep in enumerate(valid_mask.cpu().tolist())
                    if keep
                ]
                candidate_quality = {}
                candidate_quality_tensors = {}
                for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
                    values = batch.get(key)
                    if values is not None:
                        tensor_value = values[i][:candidate_limit][valid_mask].to(device=device, dtype=torch.float32)
                        candidate_quality_tensors[key] = tensor_value
                        candidate_quality[key] = tensor_value.detach().cpu().float().numpy().tolist()
            else:
                candidate_poses = pose_init[i : i + 1]
                candidate_scores = torch.tensor(
                    [float(batch["retrieval_score"][i])], device=device, dtype=torch.float32
                )
                candidate_frame_ids = [int(batch["retrieval_frame_id"][i])]
                candidate_image_names = [batch["retrieval_image_name"][i]]
                candidate_quality = {}
                candidate_quality_tensors = {}

            hyp_count = int(candidate_poses.shape[0])
            pose_gt_rep = pose_gt_i.repeat(hyp_count, 1, 1)

            init_rot_h, init_trans_h = compute_pose_errors(candidate_poses, pose_gt_rep)
            init_render_top1, _ = render_batch(
                gaussians, dcff_renderer, feat_sharp, candidate_poses[:1], K, render_h, render_w
            )
            chunk_size = int(hyp_chunk_size) if int(hyp_chunk_size) > 0 else hyp_count
            refined_pose_chunks = []
            refined_render_chunks = []
            refine_diag_chunks: Dict[str, List[torch.Tensor]] = {}
            for h_start in range(0, hyp_count, max(1, chunk_size)):
                h_end = min(hyp_count, h_start + max(1, chunk_size))
                chunk_count = h_end - h_start
                query_fine_rep = query_fine_i.repeat(chunk_count, 1, 1, 1)
                query_coarse_rep = (
                    query_coarse_i.repeat(chunk_count, 1, 1, 1) if query_coarse_i is not None else None
                )
                refine_result = refine_pose_batch(
                    model=model,
                    gaussians=gaussians,
                    dcff_renderer=dcff_renderer,
                    feat_sharp=feat_sharp,
                    query_fine=query_fine_rep,
                    query_coarse=query_coarse_rep,
                    pose_init=candidate_poses[h_start:h_end],
                    K=K,
                    render_intr=render_intr,
                    outer_iters=outer_iters,
                    solver=solver,
                    irls_iters=irls_iters,
                    robust_kernel=robust_kernel,
                    direct_refine_iters=direct_refine_iters,
                    collect_diagnostics=collect_refine_diagnostics,
                    refine_gate=gate_cfg,
                )
                if collect_refine_diagnostics:
                    refined_pose_chunk, refined_render_chunk, refine_diag_chunk = refine_result
                    for key, value in refine_diag_chunk.items():
                        refine_diag_chunks.setdefault(key, []).append(value)
                else:
                    refined_pose_chunk, refined_render_chunk = refine_result
                refined_pose_chunks.append(refined_pose_chunk)
                refined_render_chunks.append(refined_render_chunk)
            refined_poses = torch.cat(refined_pose_chunks, dim=0)
            refined_renders = torch.cat(refined_render_chunks, dim=0)
            refine_diagnostics = {
                key: torch.cat(values, dim=0)
                for key, values in refine_diag_chunks.items()
            }
            final_rot_h, final_trans_h = compute_pose_errors(refined_poses, pose_gt_rep)

            top1_final_pose = refined_poses[:1]
            top1_final_render = refined_renders[:1]
            candidate_feature_scores = None
            candidate_rgb_errors = None
            candidate_loftr_render = None
            candidate_quality_scores = compute_pnp_quality_prior(
                fallback_scores=candidate_scores,
                candidate_quality=candidate_quality_tensors,
            )
            candidate_support_counts = compute_candidate_support_counts(
                refined_poses,
                consensus_radius_m,
            )
            candidate_centers = camera_centers_from_w2c(refined_poses)
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
                    candidate_rgb_errors = rgb_errors
                else:
                    selected_idx = 0
                    candidate_rgb_errors = []
                fused_pose = refined_poses[selected_idx]
                fuse_info = {
                    "cluster_size": 1.0,
                    "num_hypotheses": float(hyp_count),
                    "anchor_idx": float(selected_idx),
                    "selected_idx": float(selected_idx),
                    "used_top1_fallback": 1.0 if selected_idx == 0 else 0.0,
                }
            elif pose_fusion == "feature_select":
                candidate_feature_scores = compute_candidate_feature_residuals(
                    query_fine_i,
                    refined_renders,
                    model=model,
                    normalize=True,
                )
                selected_idx = int(torch.argmin(candidate_feature_scores).item())
                fused_pose = refined_poses[selected_idx]
                fuse_info = {
                    "cluster_size": 1.0,
                    "num_hypotheses": float(hyp_count),
                    "anchor_idx": float(selected_idx),
                    "selected_idx": float(selected_idx),
                    "used_top1_fallback": 1.0 if selected_idx == 0 else 0.0,
                }
            elif pose_fusion in {"loftr_render_select", "loftr_render_pose"}:
                rgb_path = batch["query_rgb_path"][i]
                if rgb_path and os.path.isfile(rgb_path):
                    from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution

                    if loftr_verifier is None:
                        loftr_verifier = LoFTRInitializer(
                            device=device,
                            loftr_long_edge=int(loftr_select_long_edge),
                            confidence_threshold=float(loftr_select_conf),
                            min_matches=int(loftr_select_min_matches),
                            reproj_threshold=float(loftr_select_reproj_threshold),
                            pnp_iters=int(loftr_select_pnp_iters),
                            use_magsac=bool(loftr_select_use_magsac),
                        )
                    query_rgb_np = load_rgb_image(rgb_path)
                    orig_hw = tuple(int(v) for v in getattr(model, "IMG_HW", query_rgb_np.shape[:2]))
                    intrinsics = getattr(model, "BASE_INTRINSICS", None)
                    if intrinsics is None:
                        intrinsics = model._scale_intrinsics(orig_hw[0], orig_hw[1])
                    loftr_hw = _compute_loftr_resolution(orig_hw, int(loftr_select_long_edge))
                    candidate_indices = choose_loftr_validation_indices(
                        candidate_scores=candidate_scores,
                        candidate_quality_scores=candidate_quality_scores,
                        candidate_support_counts=candidate_support_counts,
                        topn=int(loftr_select_topn),
                    )
                    fused_pose, fuse_info, candidate_loftr_render = run_loftr_render_validation(
                        loftr_verifier=loftr_verifier,
                        gaussians=gaussians,
                        query_rgb=query_rgb_np,
                        refined_poses=refined_poses,
                        candidate_indices=candidate_indices,
                        intrinsics=intrinsics,
                        orig_hw=orig_hw,
                        loftr_hw=loftr_hw,
                        device=device,
                        use_validated_pose=(pose_fusion == "loftr_render_pose"),
                        score_mode=loftr_select_score_mode,
                    )
                else:
                    fused_pose = refined_poses[0]
                    candidate_loftr_render = {
                        "scores": [],
                        "inliers": [],
                        "matches": [],
                        "raw_matches": [],
                        "depth_valid": [],
                        "mean_conf": [],
                        "consistency_m": [],
                        "reproj_rmse": [],
                        "reproj_median": [],
                        "inlier_ratio": [],
                        "inlier_conf_mean": [],
                    }
                    fuse_info = {
                        "cluster_size": 0.0,
                        "num_hypotheses": float(hyp_count),
                        "anchor_idx": 0.0,
                        "selected_idx": 0.0,
                        "used_top1_fallback": 1.0,
                    }
            else:
                fused_pose, fuse_info = fuse_candidate_poses(
                    refined_poses,
                    method=pose_fusion,
                    retrieval_scores=candidate_scores,
                    consensus_radius_m=consensus_radius_m,
                    consensus_min_size=consensus_min_size,
                    quality_scores=candidate_quality_scores,
                    quality_consensus_large_size=quality_consensus_large_size,
                    quality_consensus_small_size=quality_consensus_small_size,
                    quality_consensus_quality_weight=quality_consensus_quality_weight,
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
            selected_idx = int(fuse_info["selected_idx"])

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
                "top1_rot_gain_deg": float(init_rot_h[0].item() - top1_final_rot[0].item()),
                "top1_trans_gain_mm": float(init_trans_h[0].item() - top1_final_trans[0].item()),
                "fused_rot_gain_deg": float(init_rot_h[0].item() - fused_rot[0].item()),
                "fused_trans_gain_mm": float(init_trans_h[0].item() - fused_trans[0].item()),
                "top1_trans_worsened": bool(float(top1_final_trans[0].item()) > float(init_trans_h[0].item())),
                "fused_trans_worsened": bool(float(fused_trans[0].item()) > float(init_trans_h[0].item())),
                "oracle_final_rot_err_deg": oracle_rot,
                "oracle_final_trans_err_mm": oracle_trans,
                "best_candidate_rank": oracle_idx,
                "best_candidate_image_name": candidate_image_names[oracle_idx] if oracle_idx < len(candidate_image_names) else "",
                "best_candidate_score": float(candidate_scores[oracle_idx].item()),
                "hyp_candidate_scores": json.dumps(candidate_scores.detach().cpu().float().numpy().tolist()),
                "hyp_candidate_quality_scores": json.dumps(
                    candidate_quality_scores.detach().cpu().float().numpy().tolist()
                ),
                "hyp_consensus_support": json.dumps(
                    candidate_support_counts.detach().cpu().float().numpy().tolist()
                ),
                "hyp_camera_centers": json.dumps(candidate_centers.detach().cpu().float().numpy().tolist()),
                "hyp_init_rot_err_deg": json.dumps(init_rot_h.detach().cpu().float().numpy().tolist()),
                "hyp_init_trans_err_mm": json.dumps(init_trans_h.detach().cpu().float().numpy().tolist()),
                "hyp_final_rot_err_deg": json.dumps(final_rot_h.detach().cpu().float().numpy().tolist()),
                "hyp_final_trans_err_mm": json.dumps(final_trans_h.detach().cpu().float().numpy().tolist()),
            }
            for key, values in candidate_quality.items():
                record[f"hyp_{key}"] = json.dumps(values)
            if candidate_feature_scores is not None:
                record["top1_feature_residual"] = float(candidate_feature_scores[0].item())
                record["selected_feature_residual"] = float(candidate_feature_scores[selected_idx].item())
                record["oracle_feature_residual"] = float(candidate_feature_scores[oracle_idx].item())
                record["feature_selected_rank"] = selected_idx
                record["hyp_feature_residuals"] = json.dumps(
                    candidate_feature_scores.detach().cpu().float().numpy().tolist()
                )
            if candidate_rgb_errors is not None:
                record["hyp_rgb_mse"] = json.dumps(candidate_rgb_errors)
            if candidate_loftr_render is not None:
                record["hyp_loftr_render_scores"] = json.dumps(candidate_loftr_render.get("scores", []))
                record["hyp_loftr_render_inliers"] = json.dumps(candidate_loftr_render.get("inliers", []))
                record["hyp_loftr_render_matches"] = json.dumps(candidate_loftr_render.get("matches", []))
                record["hyp_loftr_render_raw_matches"] = json.dumps(
                    candidate_loftr_render.get("raw_matches", [])
                )
                record["hyp_loftr_render_depth_valid"] = json.dumps(
                    candidate_loftr_render.get("depth_valid", [])
                )
                record["hyp_loftr_render_mean_conf"] = json.dumps(candidate_loftr_render.get("mean_conf", []))
                record["hyp_loftr_render_consistency_m"] = json.dumps(
                    candidate_loftr_render.get("consistency_m", [])
                )
                record["hyp_loftr_render_reproj_rmse"] = json.dumps(
                    candidate_loftr_render.get("reproj_rmse", [])
                )
                record["hyp_loftr_render_reproj_median"] = json.dumps(
                    candidate_loftr_render.get("reproj_median", [])
                )
                record["hyp_loftr_render_inlier_ratio"] = json.dumps(
                    candidate_loftr_render.get("inlier_ratio", [])
                )
                record["hyp_loftr_render_inlier_conf_mean"] = json.dumps(
                    candidate_loftr_render.get("inlier_conf_mean", [])
                )
            for key, value in refine_diagnostics.items():
                if not torch.is_tensor(value) or value.numel() == 0:
                    continue
                value_cpu = value.detach().float().cpu()
                flat_value = value_cpu.reshape(value_cpu.shape[0], -1)
                hyp_values = flat_value.mean(dim=1)
                top1_value = float(flat_value[0].mean().item())
                record[f"top1_diag_{key}"] = top1_value
                diag_record_values.setdefault(key, []).append(top1_value)
                if flat_value.shape[0] > 1:
                    record[f"selected_diag_{key}"] = float(hyp_values[selected_idx].item())
                    record[f"oracle_diag_{key}"] = float(hyp_values[oracle_idx].item())
                    record[f"hyp_diag_{key}"] = json.dumps(hyp_values.numpy().tolist())
                    record[f"hyp_diag_{key}_mean"] = float(hyp_values.mean().item())
                    record[f"hyp_diag_{key}_min"] = float(hyp_values.min().item())
                    record[f"hyp_diag_{key}_max"] = float(hyp_values.max().item())
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
        "refine_diagnostics": {
            f"{key}_mean": float(np.mean(values))
            for key, values in sorted(diag_record_values.items())
            if values
        },
    }, records


def main():
    parser = argparse.ArgumentParser(description="Real-init evaluation for concat localizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--localization_manifest",
        default=None,
        help="Manifest exported by feature_extract.export; overrides feature_dir, joint map checkpoint, and fixed init cache.",
    )
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
    parser.add_argument("--eval_split", default="test", choices=["train", "test"])
    parser.add_argument("--eval_split_file", default=None)
    parser.add_argument("--eval_start", type=int, default=0)
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
        choices=[
            "none",
            "centroid",
            "consensus",
            "consensus_centroid",
            "quality_weighted_consensus_centroid",
            "score_select",
            "quality_consensus",
            "rgb_select",
            "feature_select",
            "loftr_render_select",
            "loftr_render_pose",
        ],
        help="How to combine refined top-k hypotheses.",
    )
    parser.add_argument("--consensus_radius_m", type=float, default=1.0)
    parser.add_argument("--consensus_min_size", type=int, default=2)
    parser.add_argument("--quality_consensus_large_size", type=int, default=15)
    parser.add_argument("--quality_consensus_small_size", type=int, default=10)
    parser.add_argument("--quality_consensus_quality_weight", type=float, default=0.7)
    parser.add_argument("--loftr_select_topn", type=int, default=3)
    parser.add_argument("--loftr_select_long_edge", type=int, default=640)
    parser.add_argument("--loftr_select_conf", type=float, default=0.3)
    parser.add_argument("--loftr_select_min_matches", type=int, default=6)
    parser.add_argument("--loftr_select_reproj_threshold", type=float, default=8.0)
    parser.add_argument("--loftr_select_pnp_iters", type=int, default=2000)
    parser.add_argument("--loftr_select_use_magsac", action="store_true")
    parser.add_argument(
        "--loftr_select_score_mode",
        choices=["basic", "pnp_quality"],
        default="basic",
        help="Score LoFTR-render validated candidates using the legacy basic score or PnP reprojection quality.",
    )
    parser.add_argument(
        "--hyp_chunk_size",
        type=int,
        default=0,
        help="Refine top-k hypotheses in chunks to reduce peak memory; 0 disables chunking.",
    )
    parser.add_argument("--cascade_rounds", type=int, default=0,
                        help="Number of cascade refinement rounds after initial refinement")
    parser.add_argument("--cascade_starts", type=int, default=10,
                        help="Number of multi-start perturbations per cascade round")
    parser.add_argument("--cascade_noise_deg", type=float, default=0.3,
                        help="Rotation noise std for cascade perturbations")
    parser.add_argument("--cascade_noise_m", type=float, default=0.05,
                        help="Translation noise std for cascade perturbations")
    parser.add_argument(
        "--record_refine_diagnostics",
        action="store_true",
        help="Write flow/confidence/pose-step diagnostics for each refined sample.",
    )
    parser.add_argument("--refine_gate_max_step_trans_mm", type=float, default=0.0)
    parser.add_argument("--refine_gate_max_step_rot_deg", type=float, default=0.0)
    parser.add_argument("--refine_gate_max_delta_xi_norm", type=float, default=0.0)
    parser.add_argument("--refine_gate_max_flow_mag_mean", type=float, default=0.0)
    parser.add_argument("--refine_gate_max_flow_mag_max", type=float, default=0.0)
    parser.add_argument("--refine_gate_min_confidence_mean", type=float, default=0.0)
    parser.add_argument("--refine_gate_max_confidence_lowfrac", type=float, default=0.0)
    parser.add_argument("--refine_gate_min_depth_valid_ratio", type=float, default=0.0)
    args = parser.parse_args()
    refine_gate = build_refine_gate_config(args)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    config = load_mainline_config(args.config, localization_manifest=args.localization_manifest)

    exp_name = config.get("exp_name", Path(args.checkpoint).stem)
    out_root = args.output_dir or os.path.join(config.get("output_dir", "output"), exp_name)
    run_tag = "real_init"
    if args.retrieval_topk > 1 or args.pose_fusion != "none":
        run_tag = f"real_init_top{args.retrieval_topk}_{args.pose_fusion}"
    if args.cascade_rounds > 0:
        run_tag += f"_cascade{args.cascade_rounds}x{args.cascade_starts}"
    if refine_gate:
        run_tag += "_gated"
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
    if args.init_poses_path is None:
        manifest_val_init = config.get("dataset", {}).get("val_init_poses_path")
        manifest_init = config.get("dataset", {}).get("init_poses_path")
        init_poses_path = manifest_val_init or manifest_init or init_poses_path
    save_init_poses_path = args.save_init_poses_path or init_poses_path

    print("Building DCFF rendering pipeline...")
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    load_external_local_corr_projector(model, config, device, printer=print)
    if should_restore_pose_checkpoint_map_state(config):
        pose_ckpt = load_concat_pose_checkpoint(args.checkpoint, device)
        restored_map = apply_localization_map_state(
            dcff_renderer,
            feat_sharp,
            pose_ckpt,
            printer=print,
        )
        if restored_map:
            print(f"Restored localization map state: {', '.join(restored_map)}")
    else:
        print("Keeping configured DCFF joint checkpoint; skipping map state embedded in localization checkpoint.")

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

    eval_split_file = args.eval_split_file or ds_cfg[f"{args.eval_split}_split"]
    val_ds = RadioLocRetrievalDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        split=args.eval_split,
        split_file=eval_split_file,
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
    init_candidate_stats = summarize_init_candidate_slots(val_ds)
    effective_retrieval_topk = int(
        min(
            max(1, args.retrieval_topk),
            max(1, init_candidate_stats.get("loaded_candidate_slots", args.retrieval_topk)),
        )
    )

    eval_dataset = val_ds
    eval_start = max(0, int(args.eval_start))
    eval_stop = len(val_ds)
    if args.max_samples > 0:
        eval_stop = min(len(val_ds), eval_start + int(args.max_samples))
    eval_indices = list(range(eval_start, eval_stop))
    if eval_start > 0 or args.max_samples > 0:
        eval_dataset = Subset(val_ds, eval_indices)

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
    if args.localization_manifest:
        print(f"Localization manifest: {args.localization_manifest}")
    print(f"Query feature dir: {ds_cfg['feature_dir']}")
    if config.get("dcff", {}).get("joint_checkpoint"):
        print(f"Joint map checkpoint: {config['dcff']['joint_checkpoint']}")
    print(f"Init pose cache: {save_init_poses_path}")
    print(f"Init stats: {val_ds.init_stats}")
    print(
        f"Retrieval top-k: requested={args.retrieval_topk} effective={effective_retrieval_topk} "
        f"loaded_slots={init_candidate_stats['loaded_candidate_slots']} fusion={args.pose_fusion}"
    )
    if init_candidate_stats["loaded_candidate_slots"] > args.retrieval_topk:
        print(
            "Loaded init cache has more candidate slots than requested; "
            f"using only the first {args.retrieval_topk} valid candidates."
        )
    if args.record_refine_diagnostics or refine_gate:
        print(f"Record refine diagnostics: {args.record_refine_diagnostics or bool(refine_gate)}")
    if refine_gate:
        print(f"Refine gate: {refine_gate}")
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
        quality_consensus_large_size=args.quality_consensus_large_size,
        quality_consensus_small_size=args.quality_consensus_small_size,
        quality_consensus_quality_weight=args.quality_consensus_quality_weight,
        hyp_chunk_size=args.hyp_chunk_size,
        cascade_rounds=args.cascade_rounds,
        cascade_starts=args.cascade_starts,
        cascade_noise_deg=args.cascade_noise_deg,
        cascade_noise_m=args.cascade_noise_m,
        record_refine_diagnostics=args.record_refine_diagnostics,
        refine_gate=refine_gate,
        loftr_select_topn=args.loftr_select_topn,
        loftr_select_long_edge=args.loftr_select_long_edge,
        loftr_select_conf=args.loftr_select_conf,
        loftr_select_min_matches=args.loftr_select_min_matches,
        loftr_select_reproj_threshold=args.loftr_select_reproj_threshold,
        loftr_select_pnp_iters=args.loftr_select_pnp_iters,
        loftr_select_use_magsac=args.loftr_select_use_magsac,
        loftr_select_score_mode=args.loftr_select_score_mode,
    )

    source_counts = Counter(r["init_source"] for r in records)
    readiness = "ready" if any(not src.startswith("fallback_") for src in source_counts) else "blocked"
    blocker = ""
    if readiness != "ready":
        blocker = "No non-fallback init was produced; all samples fell back."
    init_pose_cache_report_path = (
        init_poses_path if os.path.isfile(init_poses_path) else save_init_poses_path
    )

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "localization_manifest": args.localization_manifest,
        "query_feature_dir": ds_cfg["feature_dir"],
        "joint_map_checkpoint": config.get("dcff", {}).get("joint_checkpoint"),
        "joint_override_components": config.get("dcff", {}).get("joint_override_components"),
        "output_dir": out_root,
        "epoch": ckpt_epoch,
        "solver": args.solver,
        "outer_iters": outer_iters,
        "gru_iters": gru_iters,
        "retrieval_topk": args.retrieval_topk,
        "effective_retrieval_topk": effective_retrieval_topk,
        **init_candidate_stats,
        "eval_split": args.eval_split,
        "eval_split_file": eval_split_file,
        "eval_start": eval_start,
        "eval_stop": eval_stop,
        "pose_fusion": args.pose_fusion,
        "consensus_radius_m": args.consensus_radius_m,
        "quality_consensus_large_size": args.quality_consensus_large_size,
        "quality_consensus_small_size": args.quality_consensus_small_size,
        "quality_consensus_quality_weight": args.quality_consensus_quality_weight,
        "loftr_select_topn": args.loftr_select_topn,
        "loftr_select_long_edge": args.loftr_select_long_edge,
        "loftr_select_conf": args.loftr_select_conf,
        "loftr_select_min_matches": args.loftr_select_min_matches,
        "loftr_select_reproj_threshold": args.loftr_select_reproj_threshold,
        "loftr_select_pnp_iters": args.loftr_select_pnp_iters,
        "loftr_select_use_magsac": args.loftr_select_use_magsac,
        "loftr_select_score_mode": args.loftr_select_score_mode,
        "hyp_chunk_size": args.hyp_chunk_size,
        "record_refine_diagnostics": args.record_refine_diagnostics or bool(refine_gate),
        "refine_gate": refine_gate,
        "num_samples": len(records),
        "retrieval_feature_dir": retrieval_feature_dir,
        "init_pose_cache": init_pose_cache_report_path,
        "init_cache_sha256": sha256_file_or_none(init_pose_cache_report_path),
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
        "refine_diagnostics": summaries["refine_diagnostics"],
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
        f.write(f"refine_diagnostics: {json.dumps(summaries['refine_diagnostics'])}\n")
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
