#!/usr/bin/env python3
from __future__ import annotations

"""
Concatenation-Based Localization Training Script
=================================================
End-to-end camera pose estimation using concatenated query + rendered features
(no correlation) fed through a CNN to predict optical flow + geometry solver.

Pipeline:
  1. DCFF Feature Field (pre-trained, frozen): renders 64d fine features + depth
     from a 3DGS scene at a given pose
  2. ConcatPoseNet (trainable): concatenates query + rendered features + depth
     + positional encoding, predicts flow via CNN, solves pose via geometry

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m pose_refine.train \\
        --config pose_refine/configs/concat_loc_oh_v20k_v5g_student_querymap_adapt.yaml --gpu 0

    # Resume from checkpoint
    python -m pose_refine.train --config pose_refine/configs/concat_loc_oh_v20k_v5g_student_querymap_adapt.yaml \
        --resume pose_refine/output/concat_loc_oh_v20k_v5g_student_querymap_adapt/checkpoints/latest.pth

    # Warmstart (model weights only, fresh optimizer/scheduler)
    python -m pose_refine.train --config pose_refine/configs/concat_loc_oh_v20k_v5g_student_querymap_adapt.yaml \
        --warmstart pose_refine/output/concat_loc_oh_v20k_v5g_student_querymap_adapt/checkpoints/best.pth

    # Override DCFF checkpoint
    python -m pose_refine.train --config pose_refine/configs/concat_loc_oh_v20k_v5g_student_querymap_adapt.yaml \
        --dcff_checkpoint feature_field/output/dcff_radio_oh_v10/checkpoints/best.pth
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def close(self):
            pass
from tqdm import tqdm

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.radio_loc_dataset import RadioLocDataset, collate_fn, read_colmap_cameras
from data.radio_loc_retrieval_dataset import RadioLocRetrievalDataset
from feature_field.dcff import DeferredCascadedRenderer, HybridGaussianModel, SpatialHashGrid
from pose_refine import (
    apply_pose_delta,
    build_concat_pose_model,
    feature_metric_solve,
    load_local_flow_head_weights,
    load_local_matcher_weights,
    run_model_refine_iteration,
)
from pose_refine.models.concat_pose_net import ConcatPoseNet, local_correlation
from pose_refine.utils.geometry_solver import compute_image_jacobian
from pose_refine.utils.lie_algebra import se3_exp, se3_log
from feature_field import build_dcff_runtime, intrinsics_to_K, render_feature_bundle_batch
from feature_field.runtime import _apply_dcff_postprocess
from feature_field.utils.loc_reporting import save_experiment_bundle
from feature_field.utils.project_config import load_mainline_config
from feature_field.utils.project_paths import resolve_repo_path


def camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def perturb_w2c_camera_center(
    poses_w2c: torch.Tensor,
    offsets: torch.Tensor,
    *,
    frame: str = 'camera',
) -> torch.Tensor:
    """Translate camera centres by metric offsets while preserving rotation."""
    with torch.cuda.amp.autocast(enabled=False):
        poses = poses_w2c.float()
        offsets = offsets.to(device=poses.device, dtype=poses.dtype)
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0).expand(poses.shape[0], -1)
        if offsets.shape[0] == 1 and poses.shape[0] > 1:
            offsets = offsets.expand(poses.shape[0], -1)
        if offsets.shape != (poses.shape[0], 3):
            raise ValueError(
                f'offsets must have shape (B,3), got {tuple(offsets.shape)} '
                f'for B={poses.shape[0]}'
            )

        R = poses[:, :3, :3]
        centres = camera_centers_from_w2c(poses)
        if frame == 'camera':
            offsets_world = torch.bmm(R.transpose(1, 2), offsets.unsqueeze(-1)).squeeze(-1)
        elif frame == 'world':
            offsets_world = offsets
        else:
            raise ValueError(f"Unsupported perturb frame '{frame}', expected 'camera' or 'world'")

        out = poses.clone()
        new_centres = centres + offsets_world
        out[:, :3, 3] = -torch.bmm(R, new_centres.unsqueeze(-1)).squeeze(-1)
        return out


def pose_error_tensors(
    pose_pred: torch.Tensor,
    pose_gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rotation cosine loss, rotation error in degrees, translation error in metres."""
    R_pred = pose_pred[:, :3, :3].float()
    R_gt = pose_gt[:, :3, :3].float()
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_cos_loss = 1.0 - cos_angle
    rot_err = torch.acos(cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * 180.0 / math.pi
    trans_err = torch.norm(
        camera_centers_from_w2c(pose_pred.float()) - camera_centers_from_w2c(pose_gt.float()),
        dim=1,
    )
    return rot_cos_loss, rot_err, trans_err


def has_trainable_localization_feature_path(
    model: Optional[nn.Module],
    *,
    loc_use_projection: bool,
    map_decoder_active: bool,
    map_fsm_active: bool,
) -> bool:
    """Return whether localization feature losses can reach trainable parameters."""
    if map_decoder_active or map_fsm_active:
        return True
    if not loc_use_projection or model is None:
        return False
    for module_name in ('proj_shared', 'proj_query', 'proj_render', 'cross_attn'):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        if any(param.requires_grad for param in module.parameters()):
            return True
    return False


def masked_feature_cosine_distance_per_sample(
    query_feat: torch.Tensor,
    rendered_feat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample cosine distance between query and rendered feature maps."""
    query = query_feat.float()
    rendered = rendered_feat.float()
    if query.shape[-2:] != rendered.shape[-2:]:
        query = F.interpolate(query, rendered.shape[-2:], mode='bilinear', align_corners=False)
    q = F.normalize(query, dim=1)
    r = F.normalize(rendered, dim=1)
    dist = 1.0 - (q * r).sum(dim=1, keepdim=True)
    if mask is None:
        return dist.mean(dim=(1, 2, 3))
    mask_f = mask.float()
    if mask_f.ndim == 3:
        mask_f = mask_f.unsqueeze(1)
    if mask_f.shape[-2:] != dist.shape[-2:]:
        mask_f = F.interpolate(mask_f, dist.shape[-2:], mode='nearest')
    denom = mask_f.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (dist * mask_f).sum(dim=(1, 2, 3)) / denom


def feature_metric_pose_update(
    query_feat: torch.Tensor,
    rendered_feat: torch.Tensor,
    depth: torch.Tensor,
    pose_ref: torch.Tensor,
    intrinsics: Dict[str, float],
    *,
    damping: float = 1e-3,
    valid_mask: Optional[torch.Tensor] = None,
    normalize_features: bool = True,
    update_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One differentiable feature-metric GN/WLS pose update."""
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        if query.shape[-2:] != rendered.shape[-2:]:
            query = F.interpolate(query, rendered.shape[-2:], mode='bilinear', align_corners=False)
        if normalize_features:
            query = F.normalize(query, dim=1)
            rendered = F.normalize(rendered, dim=1)
        depth_s = depth.float()
        if depth_s.ndim == 4:
            depth_s = depth_s.squeeze(1)
        if valid_mask is None:
            valid_mask = (depth_s > 0.05).unsqueeze(1).float()
        elif valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1).float()
        else:
            valid_mask = valid_mask.float()
        if valid_mask.shape[-2:] != rendered.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, rendered.shape[-2:], mode='nearest')
        delta_xi, residual = feature_metric_solve(
            query,
            rendered,
            depth_s,
            intrinsics,
            damping=damping,
            valid_mask=valid_mask,
        )
        pose_pred = apply_pose_delta(pose_ref.float(), delta_xi.float(), scale=update_scale)
        return delta_xi, pose_pred, residual


def feature_metric_pose_update_loss(
    query_feat: torch.Tensor,
    rendered_feat: torch.Tensor,
    depth: torch.Tensor,
    pose_ref: torch.Tensor,
    pose_gt: torch.Tensor,
    intrinsics: Dict[str, float],
    *,
    damping: float = 1e-3,
    valid_mask: Optional[torch.Tensor] = None,
    normalize_features: bool = True,
    update_scale: float = 1.0,
    rot_weight: float = 1.0,
    trans_weight: float = 50.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Loss that trains features to produce a useful feature-metric pose update."""
    delta_xi, pose_pred, residual = feature_metric_pose_update(
        query_feat,
        rendered_feat,
        depth,
        pose_ref,
        intrinsics,
        damping=damping,
        valid_mask=valid_mask,
        normalize_features=normalize_features,
        update_scale=update_scale,
    )
    with torch.cuda.amp.autocast(enabled=False):
        rot_cos_loss, rot_err_deg, trans_err_m = pose_error_tensors(pose_pred, pose_gt.float())
        trans_loss = trans_err_m.mean()
        rot_loss = rot_cos_loss.mean()
        loss = rot_weight * rot_loss + trans_weight * trans_loss
        delta_trans_mm = torch.linalg.norm(delta_xi[:, :3].float(), dim=1).mean() * 1000.0
        residual_mean = residual.float().abs().mean()

    return loss, {
        'fm_pose_loss': float(loss.detach().item()),
        'fm_rot_err_deg': float(rot_err_deg.detach().mean().item()),
        'fm_trans_err_mm': float((trans_err_m.detach() * 1000.0).mean().item()),
        'fm_delta_trans_mm': float(delta_trans_mm.detach().item()),
        'fm_residual_l1': float(residual_mean.detach().item()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FeatSharp / DepthGuidedRefiner must stay aligned with feature_field.runtime
# ═════════════════════════════════════════════════════════════════════════════

class FeatSharp(nn.Module):
    """Lightweight learnable sharpening applied after rasterization."""

    def __init__(self, feature_dim, kernel_size=3):
        super().__init__()
        self.sharpen = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim, kernel_size,
                padding=kernel_size // 2, groups=feature_dim,
            ),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        for m in self.sharpen:
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if m.kernel_size == (1, 1):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    m.weight.data *= 0.01

    def forward(self, x, depth=None, alpha=None):
        return x + self.sharpen(x)


class DepthGuidedRefiner(nn.Module):
    """Spatial refinement conditioned on depth and alpha."""

    def __init__(self, feature_dim=64, hidden_dim=128):
        super().__init__()
        input_dim = feature_dim + 2
        self.refiner = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_dim, 1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, features, depth=None, alpha=None):
        if depth is None or alpha is None:
            return features
        fh, fw = features.shape[-2:]
        if depth.shape[-2:] != (fh, fw):
            depth = F.interpolate(depth, (fh, fw), mode='bilinear', align_corners=False)
        if alpha.shape[-2:] != (fh, fw):
            alpha = F.interpolate(alpha, (fh, fw), mode='bilinear', align_corners=False)
        depth_norm = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        x = torch.cat([features, depth_norm, alpha], dim=1)
        return features + self.refiner(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Logging
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str, exp_name: str) -> logging.Logger:
    """Configure dual logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, f'{exp_name}_train.log'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ═════════════════════════════════════════════════════════════════════════════
#  Loss Functions
# ═════════════════════════════════════════════════════════════════════════════

def pose_loss(
    delta_xi: torch.Tensor,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    rot_loss_type: str = 'cosine',
    loss_mode: str = 'compose',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SE(3) pose refinement loss. All computation in fp32."""
    with torch.cuda.amp.autocast(enabled=False):
        delta_xi = delta_xi.float()
        pose_init = pose_init.float()
        pose_gt = pose_gt.float()

        T_rel_gt = torch.bmm(pose_gt, torch.inverse(pose_init))
        gt_xi = se3_log(T_rel_gt)

        if loss_mode == 'lie':
            pred_trans = delta_xi[:, :3]
            pred_rot = delta_xi[:, 3:]
            gt_trans = gt_xi[:, :3]
            gt_rot = gt_xi[:, 3:]
            rot_loss_val = F.smooth_l1_loss(pred_rot, gt_rot)
            trans_loss_val = F.smooth_l1_loss(pred_trans, gt_trans)
            loss = rot_loss_val * rot_weight + trans_loss_val * trans_weight
        else:
            T_delta = se3_exp(delta_xi)
            pose_pred = torch.bmm(T_delta, pose_init)

            R_pred = pose_pred[:, :3, :3]
            R_gt = pose_gt[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)

            if rot_loss_type == 'cosine':
                rot_loss_val = (1.0 - cos_angle).mean()
            else:
                # atan2(sin, cos) keeps a usable gradient for sub-degree
                # errors. Clamping cos to 1 - eps makes the loss flat below
                # about 0.8 deg, which is exactly the regime this refiner must
                # improve for Cambridge OldHospital.
                skew_vec = torch.stack(
                    [
                        R_rel[:, 2, 1] - R_rel[:, 1, 2],
                        R_rel[:, 0, 2] - R_rel[:, 2, 0],
                        R_rel[:, 1, 0] - R_rel[:, 0, 1],
                    ],
                    dim=1,
                )
                sin_angle = 0.5 * torch.linalg.vector_norm(skew_vec, dim=1)
                rot_loss_val = torch.atan2(sin_angle, cos_angle).mean()

            c_pred = camera_centers_from_w2c(pose_pred)
            c_gt = camera_centers_from_w2c(pose_gt)
            trans_loss_val = torch.norm(c_pred - c_gt, dim=1).mean()
            loss = rot_loss_val * rot_weight + trans_loss_val * trans_weight

        # Metrics always stay in composed pose space.
        T_delta = se3_exp(delta_xi)
        pose_pred = torch.bmm(T_delta, pose_init)
        R_pred = pose_pred[:, :3, :3]
        R_gt = pose_gt[:, :3, :3]
        R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)

        cos_for_metric = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        rot_err_deg = torch.acos(cos_for_metric) * 180.0 / math.pi

        c_pred = camera_centers_from_w2c(pose_pred)
        c_gt = camera_centers_from_w2c(pose_gt)
        trans_err = torch.norm(c_pred - c_gt, dim=1)

    rot_mean = rot_err_deg.mean().item()
    trans_mean = (trans_err * 1000).mean().item()
    if not math.isfinite(rot_mean):
        rot_mean = float('nan')
    if not math.isfinite(trans_mean):
        trans_mean = float('nan')
    return loss, {
        'rot_err_deg': rot_mean,
        'trans_err_mm': trans_mean,
        'pose_loss': loss.item(),
    }


def flow_loss_fn(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    huber_delta: float = 5.0,
    weight_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Masked Huber flow loss with EPE metric.

    Args:
        flow_pred: (B, 2, H, W) predicted flow
        flow_gt:   (B, 2, H, W) GT flow
        valid_mask: (B, 1, H, W) validity mask
        weight_map: optional (B, 1, H, W) per-pixel weights. The map is
            normalized over valid pixels to keep the loss scale comparable.
        huber_delta: Huber threshold in pixels
    """
    diff = flow_pred - flow_gt
    abs_diff = diff.abs()
    loss_map = torch.where(
        abs_diff <= huber_delta,
        0.5 * diff.pow(2) / huber_delta,
        abs_diff - 0.5 * huber_delta,
    )
    metrics: Dict[str, float] = {}
    if weight_map is not None:
        weight_map = weight_map.to(device=loss_map.device, dtype=loss_map.dtype)
        if weight_map.shape[-2:] != loss_map.shape[-2:]:
            weight_map = F.interpolate(
                weight_map,
                size=loss_map.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        if weight_map.shape[1] != 1:
            weight_map = weight_map.mean(dim=1, keepdim=True)
        raw_mean = (weight_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        weight_map = weight_map / raw_mean.clamp(min=1e-6)
        loss_map = loss_map * weight_map
        metrics['flow_weight_mean'] = raw_mean.item()
        metrics['flow_weight_max'] = weight_map.max().item()

    n_valid = valid_mask.sum().clamp(min=1.0)
    loss = (loss_map * valid_mask).sum() / (n_valid * 2)

    # EPE metric
    epe_map = torch.norm(diff, dim=1, keepdim=True)
    epe = (epe_map * valid_mask).sum() / n_valid

    metrics.update({
        'flow_loss': loss.item(),
        'flow_epe': epe.item(),
    })
    return loss, metrics


def compute_observability_flow_weight(
    depth: torch.Tensor,
    target_hw: Tuple[int, int],
    intrinsics: Dict[str, float],
    valid_mask: Optional[torch.Tensor] = None,
    mode: str = 'rot',
    strength: float = 1.0,
    max_weight: float = 6.0,
) -> torch.Tensor:
    """Build a normalized per-pixel flow-loss weight from image Jacobians.

    This emphasizes pixels whose flow residuals are more informative for the
    requested pose components. It is intentionally normalized to mean 1 over
    valid pixels so enabling it changes emphasis, not the effective LR.
    """
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    B, H, W = depth.shape
    tH, tW = target_hw
    if (tH, tW) != (H, W):
        sx, sy = tW / W, tH / H
        depth_for_jac = F.interpolate(
            depth.unsqueeze(1),
            size=(tH, tW),
            mode='nearest',
        ).squeeze(1)
        jac_intr = dict(intrinsics)
        jac_intr['fx'] = intrinsics['fx'] * sx
        jac_intr['fy'] = intrinsics['fy'] * sy
        jac_intr['cx'] = intrinsics['cx'] * sx
        jac_intr['cy'] = intrinsics['cy'] * sy
    else:
        depth_for_jac = depth
        jac_intr = intrinsics

    Ju, Jv, jac_valid = compute_image_jacobian(depth_for_jac.float(), jac_intr)
    mode = str(mode).lower()
    if mode in {'yaw', 'wz', 'zrot'}:
        cols = slice(5, 6)
    elif mode in {'rot', 'rotation'}:
        cols = slice(3, 6)
    elif mode in {'trans', 'translation'}:
        cols = slice(0, 3)
    elif mode in {'pose', 'all', 'balanced'}:
        cols = slice(0, 6)
    else:
        raise ValueError(
            f"Unsupported observability flow-weight mode '{mode}'. "
            "Expected yaw, rot, trans, or pose."
        )

    signal = torch.sqrt(
        Ju[:, :, cols].pow(2).sum(dim=(2,))
        + Jv[:, :, cols].pow(2).sum(dim=(2,))
        + 1e-12
    ).reshape(B, 1, tH, tW)

    if valid_mask is None:
        valid = jac_valid.reshape(B, 1, tH, tW).to(signal.dtype)
    else:
        valid = valid_mask.to(device=signal.device, dtype=signal.dtype)
        if valid.shape[-2:] != (tH, tW):
            valid = F.interpolate(valid, size=(tH, tW), mode='nearest')
        valid = valid * jac_valid.reshape(B, 1, tH, tW).to(signal.dtype)

    valid_sum = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    signal_mean = (signal * valid).sum(dim=(2, 3), keepdim=True) / valid_sum
    signal_norm = signal / signal_mean.clamp(min=1e-6)
    weights = 1.0 + float(strength) * signal_norm
    if max_weight and max_weight > 0:
        weights = weights.clamp(max=float(max_weight))
    weight_mean = (weights * valid).sum(dim=(2, 3), keepdim=True) / valid_sum
    weights = weights / weight_mean.clamp(min=1e-6)
    return weights * valid


def confidence_regularization_loss(
    confidence: torch.Tensor,
    target_range: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Prevent confidence collapse by penalizing extremes."""
    mean_conf = confidence.mean()
    lo, hi = target_range
    loss = torch.tensor(0.0, device=confidence.device)
    if mean_conf < lo:
        loss = (lo - mean_conf) ** 2
    elif mean_conf > hi:
        loss = (mean_conf - hi) ** 2
    return loss, {
        'conf_mean': mean_conf.item(),
        'conf_reg_loss': loss.item(),
    }


def confidence_nll_loss(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Self-supervised confidence via Laplacian NLL.

    Models flow error as Laplace(0, b) with b = 1/conf:
        NLL = |error| * conf - log(conf)
    Optimal conf* = 1/|error|, so confidence is high where flow is accurate.

    Args:
        flow_pred: (B, 2, H, W) predicted flow
        flow_gt: (B, 2, H, W) ground truth flow
        confidence: (B, 2, H, W) predicted confidence (sigmoid, 0-1)
        valid: (B, H, W) or (B, 1, H, W) valid mask
    Returns:
        loss, metrics dict
    """
    # Per-channel absolute error
    error = (flow_pred - flow_gt).abs()  # (B, 2, H, W)

    # Clamp confidence to avoid log(0)
    conf = confidence.clamp(min=1e-4, max=1.0 - 1e-4)

    # Laplacian NLL: |error| * conf - log(conf)
    # conf acts as precision (1/scale), higher conf = tighter distribution
    nll = error * conf - torch.log(conf)

    # Mask
    if valid.ndim == 3:
        mask = valid.unsqueeze(1).expand_as(nll).float()
    else:
        mask = valid.expand_as(nll).float()

    n_valid = mask.sum().clamp(min=1.0)
    loss = (nll * mask).sum() / n_valid

    # Metrics
    with torch.no_grad():
        conf_mean = confidence.mean().item()
        conf_std = confidence.std().item()
        # Correlation between confidence and inverse error
        err_flat = (error * mask).sum(dim=1).reshape(-1)
        conf_flat = (conf[:, :1] * mask[:, :1]).sum(dim=1).reshape(-1)
        # Simple correlation metric (higher = more aligned)
        high_conf_mask = conf_flat > conf_flat.median()
        if high_conf_mask.sum() > 0:
            err_high_conf = err_flat[high_conf_mask].mean().item()
            err_low_conf = err_flat[~high_conf_mask].mean().item()
        else:
            err_high_conf = err_low_conf = 0.0

    return loss, {
        'conf_nll_loss': loss.item(),
        'conf_mean': conf_mean,
        'conf_std': conf_std,
        'conf_err_ratio': err_low_conf / max(err_high_conf, 1e-6),
    }


def confidence_validity_loss(
    confidence: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervise solver confidence to suppress pixels without GT correspondences."""
    with torch.cuda.amp.autocast(enabled=False):
        if valid.ndim == 3:
            target = valid.unsqueeze(1).float()
        else:
            target = valid.float()

        conf = confidence.float()
        if conf.shape[1] > 1:
            conf = conf.mean(dim=1, keepdim=True)
        conf = conf.clamp(min=1e-4, max=1.0 - 1e-4)
        if target.shape[-2:] != conf.shape[-2:]:
            target = F.interpolate(target, conf.shape[-2:], mode='nearest')

        pos = target > 0.5
        neg = ~pos
        pos_loss = -torch.log(conf[pos]).mean() if pos.any() else conf.new_tensor(0.0)
        neg_loss = -torch.log(1.0 - conf[neg]).mean() if neg.any() else conf.new_tensor(0.0)
        loss = 0.5 * (pos_loss + neg_loss)
        with torch.no_grad():
            conf_valid = conf[pos].mean().item() if pos.any() else 0.0
            conf_invalid = conf[neg].mean().item() if neg.any() else 0.0

    return loss, {
        'conf_valid_loss': loss.item(),
        'conf_valid_mean': conf_valid,
        'conf_invalid_mean': conf_invalid,
    }


def confidence_epe_bce_loss(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    good_px: float = 1.0,
    bad_px: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calibrate confidence from endpoint error thresholds.

    Pixels below ``good_px`` are positive, pixels above ``bad_px`` are
    negative, and the band in between is ignored. This gives the WLS confidence
    head a direct signal to suppress unreliable local matches.
    """
    with torch.cuda.amp.autocast(enabled=False):
        epe = torch.norm((flow_pred - flow_gt).float(), dim=1, keepdim=True)
        conf = confidence.float()
        if conf.shape[1] > 1:
            conf = conf.mean(dim=1, keepdim=True)
        conf = conf.clamp(min=1e-4, max=1.0 - 1e-4)

        if valid.ndim == 3:
            mask = valid.unsqueeze(1).float()
        else:
            mask = valid.float()
        if mask.shape[-2:] != conf.shape[-2:]:
            mask = F.interpolate(mask, conf.shape[-2:], mode='nearest')
        if epe.shape[-2:] != conf.shape[-2:]:
            epe = F.interpolate(epe, conf.shape[-2:], mode='bilinear', align_corners=False)

        pos = (epe <= float(good_px)) & (mask > 0.5)
        neg = (epe >= float(bad_px)) & (mask > 0.5)
        used = pos | neg
        if used.any():
            target = pos.float()
            loss_map = F.binary_cross_entropy(conf, target, reduction='none')
            loss = loss_map[used].mean()
        else:
            loss = conf.new_tensor(0.0)

        with torch.no_grad():
            target_mean = pos.float()[used].mean().item() if used.any() else 0.0
            pos_mean = conf[pos].mean().item() if pos.any() else 0.0
            neg_mean = conf[neg].mean().item() if neg.any() else 0.0
            used_frac = used.float().mean().item()

    return loss, {
        'conf_epe_bce_loss': loss.item(),
        'conf_epe_target_mean': target_mean,
        'conf_epe_pos_mean': pos_mean,
        'conf_epe_neg_mean': neg_mean,
        'conf_epe_used_frac': used_frac,
    }


def flow_correspondence_feature_loss(
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align rendered features with query features at GT-flow correspondences."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query = query_feat.float()
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        B, _, H, W = rendered.shape
        if query.shape[-2:] != (H, W):
            src_h, src_w = query.shape[-2:]
            query = F.interpolate(query, (H, W), mode='bilinear', align_corners=False)
            flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / src_w
            flow[:, 1] *= H / src_h
            valid = F.interpolate(valid, (H, W), mode='nearest')

        y, x = torch.meshgrid(
            torch.arange(H, device=rendered.device, dtype=torch.float32),
            torch.arange(W, device=rendered.device, dtype=torch.float32),
            indexing='ij',
        )
        sample_x = x.unsqueeze(0) + flow[:, 0]
        sample_y = y.unsqueeze(0) + flow[:, 1]
        grid_x = sample_x / max(W - 1, 1) * 2.0 - 1.0
        grid_y = sample_y / max(H - 1, 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        query_warp = F.grid_sample(
            query,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        q = F.normalize(query_warp, dim=1)
        r = F.normalize(rendered, dim=1)
        cos = (q * r).sum(dim=1, keepdim=True)

        denom = valid.sum().clamp(min=1.0)
        loss = ((1.0 - cos) * valid).sum() / denom
        with torch.no_grad():
            cos_mean = (cos * valid).sum() / denom
            valid_ratio = valid.mean()

    return loss, {
        'corr_feat_loss': loss.item(),
        'corr_feat_cos': cos_mean.item(),
        'corr_feat_valid': valid_ratio.item(),
    }


def local_correlation_ce_loss(
    model: ConcatPoseNet,
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Classify the GT local offset in the rendered-centered correlation window."""
    B, _, H, W = rendered_feat.shape
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 4:
            valid = valid.squeeze(1)

        if query.shape[-2:] != (H, W):
            src_h, src_w = query.shape[-2:]
            query = F.interpolate(query, (H, W), mode='bilinear', align_corners=False)
            flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / src_w
            flow[:, 1] *= H / src_h
            valid = F.interpolate(valid.unsqueeze(1), (H, W), mode='nearest').squeeze(1)

        q_proj, r_proj = model._project_for_local_corr(query, rendered)
        corr = local_correlation(r_proj, q_proj, radius=radius)
        dx = torch.round(flow[:, 0]).long()
        dy = torch.round(flow[:, 1]).long()
        in_window = (
            (dx >= -radius) & (dx <= radius)
            & (dy >= -radius) & (dy <= radius)
            & (valid > 0.5)
        )
        target = ((dy + radius) * (2 * radius + 1) + (dx + radius)).clamp(
            0, (2 * radius + 1) ** 2 - 1,
        )
        loss_map = F.cross_entropy(
            corr.float() / max(float(temperature), 1e-6),
            target,
            reduction='none',
        )
        denom = in_window.float().sum().clamp(min=1.0)
        loss = (loss_map * in_window.float()).sum() / denom

        with torch.no_grad():
            pred = corr.argmax(dim=1)
            acc = ((pred == target) & in_window).float().sum() / denom
            coverage = in_window.float().mean()

    return loss, {
        'corr_ce_loss': loss.item(),
        'corr_ce_acc': acc.item(),
        'corr_ce_cov': coverage.item(),
    }


def local_correlation_subpixel_ce_loss_from_corr(
    corr: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Soft cross-entropy with bilinear subpixel targets in the local window."""
    with torch.cuda.amp.autocast(enabled=False):
        corr_f = corr.float()
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        B, channels, H, W = corr_f.shape
        window = 2 * radius + 1
        expected_channels = window * window
        if channels != expected_channels:
            raise ValueError(
                f'corr has {channels} channels, expected {expected_channels} '
                f'for radius={radius}'
            )
        if flow.shape[-2:] != (H, W):
            src_h, src_w = flow.shape[-2:]
            flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / src_w
            flow[:, 1] *= H / src_h
            valid = F.interpolate(valid, (H, W), mode='nearest')
        elif valid.shape[-2:] != (H, W):
            valid = F.interpolate(valid, (H, W), mode='nearest')

        log_probs = F.log_softmax(corr_f / max(float(temperature), 1e-6), dim=1)
        fx = flow[:, 0:1]
        fy = flow[:, 1:2]
        x0 = torch.floor(fx)
        y0 = torch.floor(fy)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = (fx - x0).clamp(0.0, 1.0)
        wy1 = (fy - y0).clamp(0.0, 1.0)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        loss_map = torch.zeros(B, 1, H, W, device=corr_f.device, dtype=corr_f.dtype)
        target_mass = torch.zeros_like(loss_map)
        for yy, wy in ((y0, wy0), (y1, wy1)):
            for xx, wx in ((x0, wx0), (x1, wx1)):
                in_bounds = (
                    (valid > 0.5)
                    & (xx >= -radius)
                    & (xx <= radius)
                    & (yy >= -radius)
                    & (yy <= radius)
                )
                weight = (wx * wy) * in_bounds.float()
                idx = ((yy.long() + radius) * window + (xx.long() + radius)).clamp(
                    0,
                    expected_channels - 1,
                )
                gathered = log_probs.gather(1, idx)
                loss_map = loss_map - weight * gathered
                target_mass = target_mass + weight

        in_window = ((valid > 0.5) & (target_mass > 1e-6)).float()
        denom = in_window.sum().clamp(min=1.0)
        loss = ((loss_map / target_mass.clamp(min=1e-6)) * in_window).sum() / denom

        with torch.no_grad():
            offsets = torch.arange(-radius, radius + 1, device=corr_f.device, dtype=corr_f.dtype)
            dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
            dx = dx.reshape(1, expected_channels, 1, 1)
            dy = dy.reshape(1, expected_channels, 1, 1)
            probs = torch.softmax(corr_f / max(float(temperature), 1e-6), dim=1)
            pred_flow = torch.cat([
                (probs * dx).sum(dim=1, keepdim=True),
                (probs * dy).sum(dim=1, keepdim=True),
            ], dim=1)
            epe_map = torch.norm(pred_flow - flow, dim=1, keepdim=True)
            epe = (epe_map * in_window).sum() / denom
            nearest_dx = torch.round(fx).long()
            nearest_dy = torch.round(fy).long()
            nearest_target = ((nearest_dy + radius) * window + (nearest_dx + radius)).clamp(
                0,
                expected_channels - 1,
            )
            pred = corr_f.argmax(dim=1, keepdim=True)
            acc = ((pred == nearest_target) & (in_window > 0.5)).float().sum() / denom
            coverage = in_window.mean()

    return loss, {
        'corr_subpx_ce_loss': loss.item(),
        'corr_subpx_flow_epe': epe.item(),
        'corr_subpx_acc': acc.item(),
        'corr_subpx_cov': coverage.item(),
    }


def local_correlation_subpixel_ce_loss(
    model: ConcatPoseNet,
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train projected local correlation with a bilinear subpixel target."""
    B, _, H, W = rendered_feat.shape
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 4:
            valid = valid.squeeze(1)

        if query.shape[-2:] != (H, W):
            src_h, src_w = query.shape[-2:]
            query = F.interpolate(query, (H, W), mode='bilinear', align_corners=False)
            flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / src_w
            flow[:, 1] *= H / src_h
            valid = F.interpolate(valid.unsqueeze(1), (H, W), mode='nearest').squeeze(1)

        q_proj, r_proj = model._project_for_local_corr(query, rendered)
        corr = local_correlation(r_proj, q_proj, radius=radius)

    return local_correlation_subpixel_ce_loss_from_corr(
        corr,
        flow,
        valid,
        radius=radius,
        temperature=temperature,
    )


def local_correlation_soft_flow_loss_from_corr(
    corr: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    temperature: float = 0.1,
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Regress subpixel flow from a rendered-centered local correlation volume."""
    with torch.cuda.amp.autocast(enabled=False):
        corr_f = corr.float()
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        B, channels, H, W = corr_f.shape
        window = 2 * radius + 1
        expected_channels = window * window
        if channels != expected_channels:
            raise ValueError(
                f'corr has {channels} channels, expected {expected_channels} '
                f'for radius={radius}'
            )
        if flow.shape[-2:] != (H, W):
            src_h, src_w = flow.shape[-2:]
            flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / src_w
            flow[:, 1] *= H / src_h
            valid = F.interpolate(valid, (H, W), mode='nearest')
        elif valid.shape[-2:] != (H, W):
            valid = F.interpolate(valid, (H, W), mode='nearest')

        offsets = torch.arange(-radius, radius + 1, device=corr_f.device, dtype=corr_f.dtype)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        dx = dx.reshape(1, expected_channels, 1, 1)
        dy = dy.reshape(1, expected_channels, 1, 1)

        weights = torch.softmax(corr_f / max(float(temperature), 1e-6), dim=1)
        pred_flow = torch.cat([
            (weights * dx).sum(dim=1, keepdim=True),
            (weights * dy).sum(dim=1, keepdim=True),
        ], dim=1)

        in_window = (
            (valid > 0.5)
            & (flow[:, :1] >= -radius)
            & (flow[:, :1] <= radius)
            & (flow[:, 1:2] >= -radius)
            & (flow[:, 1:2] <= radius)
        ).float()

        diff = pred_flow - flow
        abs_diff = diff.abs()
        loss_map = torch.where(
            abs_diff <= huber_delta,
            0.5 * diff.pow(2) / max(float(huber_delta), 1e-6),
            abs_diff - 0.5 * huber_delta,
        )
        denom = (in_window.sum() * 2.0).clamp(min=1.0)
        loss = (loss_map * in_window).sum() / denom

        with torch.no_grad():
            pixel_denom = in_window.sum().clamp(min=1.0)
            epe_map = torch.norm(diff, dim=1, keepdim=True)
            epe = (epe_map * in_window).sum() / pixel_denom
            coverage = in_window.mean()

    return loss, {
        'corr_flow_loss': loss.item(),
        'corr_flow_epe': epe.item(),
        'corr_flow_cov': coverage.item(),
    }


def local_correlation_soft_flow_loss(
    model: ConcatPoseNet,
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train the model's projected local correlation to encode subpixel flow."""
    B, _, H, W = rendered_feat.shape
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 4:
            valid = valid.squeeze(1)

        if query.shape[-2:] != (H, W):
            src_h, src_w = query.shape[-2:]
            query = F.interpolate(query, (H, W), mode='bilinear', align_corners=False)
            flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / src_w
            flow[:, 1] *= H / src_h
            valid = F.interpolate(valid.unsqueeze(1), (H, W), mode='nearest').squeeze(1)

        q_proj, r_proj = model._project_for_local_corr(query, rendered)
        corr = local_correlation(r_proj, q_proj, radius=radius)

    return local_correlation_soft_flow_loss_from_corr(
        corr,
        flow,
        valid,
        radius=radius,
        temperature=temperature,
    )


def feature_cosine_distance_per_sample(
    query_feat: torch.Tensor,
    rendered_feat: torch.Tensor,
) -> torch.Tensor:
    """Per-sample cosine distance between query and rendered feature maps."""
    return masked_feature_cosine_distance_per_sample(query_feat, rendered_feat)


def resolve_best_metric_value(
    metrics: Dict[str, float],
    metric_name: str,
) -> Tuple[float, bool, str]:
    """Return metric value, comparison direction, and resolved metric key."""
    aliases = {
        'trans': 'val_trans_median',
        'trans_median': 'val_trans_median',
        'rot': 'val_rot_median',
        'rot_median': 'val_rot_median',
        'joint_1deg_50mm': 'val_joint_1deg_50mm',
        'joint_5deg_100mm': 'val_joint_5deg_100mm',
    }
    metric_key = aliases.get(str(metric_name), str(metric_name))
    value = float(metrics.get(metric_key, float('-inf')))
    higher_is_better = metric_key.startswith('val_joint_') or metric_key.startswith('val_pct_')
    if not higher_is_better and metric_key not in metrics:
        value = float('inf')
    return value, higher_is_better, metric_key


def build_eval_sweep_record(
    *,
    outer_iters: int,
    gru_iters: int,
    metrics: Dict[str, float],
    seed_count: int = 1,
) -> Dict[str, float | int]:
    """Compact JSON-friendly record for pose-refine eval sweeps."""
    return {
        'outer_iters': int(outer_iters),
        'gru_iters': int(gru_iters),
        'seed_count': int(seed_count),
        'rot_median_deg': float(metrics.get('val_rot_median', 0.0)),
        'trans_median_mm': float(metrics.get('val_trans_median', 0.0)),
        'pct_1deg': float(metrics.get('val_pct_1deg', 0.0)),
        'joint_1deg_50mm': float(metrics.get('val_joint_1deg_50mm', 0.0)),
    }


def build_eval_sample_record(
    *,
    image_id: int,
    image_name: str = '',
    outer_iters: int,
    gru_iters: int,
    seed: int,
    init_rot_deg: float,
    init_trans_mm: float,
    one_rot_deg: float,
    one_trans_mm: float,
    final_rot_deg: float,
    final_trans_mm: float,
    flow_epe_px: float,
) -> Dict[str, float | int | str]:
    """JSON-friendly per-sample localization eval record."""
    return {
        'image_id': int(image_id),
        'image_name': str(image_name),
        'outer_iters': int(outer_iters),
        'gru_iters': int(gru_iters),
        'seed': int(seed),
        'init_rot_deg': float(init_rot_deg),
        'init_trans_mm': float(init_trans_mm),
        'one_rot_deg': float(one_rot_deg),
        'one_trans_mm': float(one_trans_mm),
        'final_rot_deg': float(final_rot_deg),
        'final_trans_mm': float(final_trans_mm),
        'flow_epe_px': float(flow_epe_px),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Visualization Utilities
# ═════════════════════════════════════════════════════════════════════════════

def features_to_pca_rgb(features: torch.Tensor) -> torch.Tensor:
    """Convert (C, H, W) feature map to (3, H, W) RGB via PCA."""
    from sklearn.decomposition import PCA
    C, H, W = features.shape
    feat_np = features.detach().cpu().float().numpy().reshape(C, -1).T
    pca = PCA(n_components=3)
    proj = pca.fit_transform(feat_np)
    for i in range(3):
        lo, hi = np.percentile(proj[:, i], [2, 98])
        proj[:, i] = np.clip((proj[:, i] - lo) / max(hi - lo, 1e-8), 0, 1)
    return torch.from_numpy(proj.T.reshape(3, H, W)).float()


# ═════════════════════════════════════════════════════════════════════════════
#  ConcatLocTrainer
# ═════════════════════════════════════════════════════════════════════════════

class ConcatLocTrainer:
    """End-to-end localization trainer: concatenated RADIO query + DCFF map."""

    def __init__(
        self,
        config: dict,
        gpu: int = 0,
        resume_path: str = None,
        warmstart_path: str = None,
        dcff_checkpoint: str = None,
    ):
        self.config = config
        self.gpu = gpu
        self.device = torch.device(f'cuda:{gpu}')

        # Output directories
        exp_name = config['exp_name']
        self.exp_name = exp_name
        output_base = resolve_repo_path(config.get('output_dir', 'output'))
        assert output_base is not None
        self.output_dir = Path(output_base) / exp_name
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.vis_dir = self.output_dir / 'vis'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(str(self.output_dir), exp_name)
        self.logger.info(f'Experiment: {exp_name}')
        self.logger.info(f'Output: {self.output_dir}')
        self.logger.info(f'Device: {self.device}')

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Override DCFF checkpoint from CLI
        if dcff_checkpoint:
            config.setdefault('dcff', {})['checkpoint'] = dcff_checkpoint

        # Training config
        tc = config.get('training', {})
        self.total_epochs = tc.get('epochs', 200)
        self.grad_clip = tc.get('grad_clip', 1.0)
        self.val_every = tc.get('val_every', 5)
        self.save_every = tc.get('save_every', 10)
        self.save_epoch_checkpoints = tc.get('save_epoch_checkpoints', False)
        self.vis_every = tc.get('vis_every', 10)
        self.num_vis_samples = tc.get('num_vis_samples', 4)
        self.use_amp = bool(tc.get('use_amp', True))
        self.max_train_batches = int(tc.get('max_train_batches', 0) or 0)
        self.max_val_batches = int(tc.get('max_val_batches', 0) or 0)
        self.feature_only_train = bool(tc.get('feature_only_train', False))
        self.warmstart_skip_prefixes = [
            str(prefix) for prefix in tc.get('warmstart_skip_prefixes', [])
        ]

        # Loss config
        loss_cfg = tc.get('loss', {})
        self.pose_weight = loss_cfg.get('pose_weight', 0.5)
        self.flow_weight = loss_cfg.get('flow_weight', 1.0)
        self.init_flow_weight = float(loss_cfg.get('init_flow_weight', 0.0))
        self.flow_observability_weight = float(loss_cfg.get('flow_observability_weight', 0.0))
        self.flow_observability_mode = str(loss_cfg.get('flow_observability_mode', 'rot'))
        self.flow_observability_max_weight = float(loss_cfg.get('flow_observability_max_weight', 6.0))
        self.rot_weight = loss_cfg.get('rot_weight', 1.0)
        self.trans_weight = loss_cfg.get('trans_weight', 10.0)
        self.rot_loss_type = loss_cfg.get('rot_loss_type', 'cosine')
        self.pose_loss_mode = loss_cfg.get('pose_loss_mode', 'compose')
        self.conf_reg_weight = loss_cfg.get('conf_reg_weight', 0.01)
        self.conf_nll_weight = loss_cfg.get('conf_nll_weight', 0.0)
        self.conf_valid_weight = loss_cfg.get('conf_valid_weight', 0.0)
        self.conf_epe_bce_weight = float(loss_cfg.get('conf_epe_bce_weight', 0.0))
        self.conf_epe_good_px = float(loss_cfg.get('conf_epe_good_px', 1.0))
        self.conf_epe_bad_px = float(loss_cfg.get('conf_epe_bad_px', 5.0))
        self.feat_match_weight = loss_cfg.get('feat_match_weight', 0.0)
        self.corr_feat_weight = loss_cfg.get('corr_feat_weight', 0.0)
        self.corr_ce_weight = loss_cfg.get('corr_ce_weight', 0.0)
        self.corr_ce_temperature = loss_cfg.get('corr_ce_temperature', 0.1)
        self.corr_subpixel_ce_weight = loss_cfg.get('corr_subpixel_ce_weight', 0.0)
        self.corr_subpixel_ce_temperature = loss_cfg.get(
            'corr_subpixel_ce_temperature',
            self.corr_ce_temperature,
        )
        self.corr_flow_weight = loss_cfg.get('corr_flow_weight', 0.0)
        self.corr_flow_temperature = loss_cfg.get(
            'corr_flow_temperature',
            self.corr_ce_temperature,
        )
        self.use_direct_trans_loss = loss_cfg.get('direct_trans_loss', True)
        self.coarse_pose_weight = loss_cfg.get('coarse_pose_weight', self.pose_weight)
        self.full_pose_weight = loss_cfg.get('full_pose_weight', 0.0)
        self.pose_rank_weight = float(loss_cfg.get('pose_rank_weight', 0.0))
        self.pose_rank_margin = float(loss_cfg.get('pose_rank_margin', 0.05))
        self.pose_rank_rot_deg = float(loss_cfg.get('pose_rank_rot_deg', 2.0))
        self.pose_rank_trans_m = float(loss_cfg.get('pose_rank_trans_m', 0.25))
        self.loc_rank_weight = float(loss_cfg.get('loc_rank_weight', 0.0))
        loc_rank_distances = loss_cfg.get('loc_rank_distances_m', [0.01, 0.02, 0.05])
        if isinstance(loc_rank_distances, (int, float)):
            loc_rank_distances = [float(loc_rank_distances)]
        self.loc_rank_distances_m = [float(v) for v in loc_rank_distances]
        self.loc_rank_negatives = max(1, int(loss_cfg.get('loc_rank_negatives', 3)))
        self.loc_rank_margin = float(loss_cfg.get('loc_rank_margin', 0.02))
        self.loc_rank_margin_per_m = float(loss_cfg.get('loc_rank_margin_per_m', 0.0))
        self.loc_rank_frame = str(loss_cfg.get('loc_rank_frame', 'camera'))
        self.loc_rank_hard_mining = bool(loss_cfg.get('loc_rank_hard_mining', True))
        self.loc_detach_query = bool(loss_cfg.get('loc_detach_query', True))
        self.loc_use_projection = bool(loss_cfg.get('loc_use_projection', False))
        self.feature_metric_weight = float(loss_cfg.get('feature_metric_weight', 0.0))
        self.feature_metric_damping = float(loss_cfg.get('feature_metric_damping', 1e-3))
        self.feature_metric_normalize = bool(loss_cfg.get('feature_metric_normalize', True))
        self.feature_metric_detach_query = bool(loss_cfg.get('feature_metric_detach_query', True))
        self.feature_metric_rot_weight = float(loss_cfg.get('feature_metric_rot_weight', 1.0))
        self.feature_metric_trans_weight = float(loss_cfg.get('feature_metric_trans_weight', 50.0))
        self.feature_metric_update_scale = float(loss_cfg.get('feature_metric_update_scale', 1.0))
        self.feature_metric_eval = bool(loss_cfg.get('feature_metric_eval', self.feature_metric_weight > 0.0))

        # Phases
        self.phase1_epochs = tc.get('phase1_epochs', 30)
        self.map_finetune_start_epoch = int(tc.get('map_finetune_start_epoch', self.phase1_epochs))
        self.fsm_finetune_start_epoch = int(tc.get('fsm_finetune_start_epoch', self.map_finetune_start_epoch))
        self.pose_rank_start_epoch = int(tc.get('pose_rank_start_epoch', self.map_finetune_start_epoch))
        self.loc_rank_start_epoch = int(loss_cfg.get(
            'loc_rank_start_epoch',
            tc.get('loc_rank_start_epoch', self.map_finetune_start_epoch),
        ))
        self.feature_metric_start_epoch = int(loss_cfg.get(
            'feature_metric_start_epoch',
            tc.get('feature_metric_start_epoch', self.map_finetune_start_epoch),
        ))
        self._map_decoder_active = False
        self._map_fsm_active = False

        # Outer iterations
        self.outer_iters_train = tc.get('outer_iters_train', 1)
        self.outer_iters_val = tc.get('outer_iters_val', 5)

        # Noise curriculum
        nc = tc.get('noise_curriculum', {})
        self.noise_rot_start = nc.get('rot_start_deg', 2.0)
        self.noise_rot_end = nc.get('rot_end_deg', 10.0)
        self.noise_trans_start = nc.get('trans_start_m', 0.05)
        self.noise_trans_end = nc.get('trans_end_m', 0.50)
        self.noise_warmup_epochs = nc.get('warmup_epochs', 60)
        # Per-sample noise range: randomize noise magnitude per sample
        self.noise_rot_min = nc.get('rot_min_deg', None)
        self.noise_trans_min = nc.get('trans_min_m', None)

        # Optional separate val noise (defaults to noise_end values)
        self.val_noise_deg = tc.get('val_noise_deg', self.noise_rot_end)
        self.val_noise_m = tc.get('val_noise_m', self.noise_trans_end)
        self.val_seed = int(tc.get('val_seed', 12345))

        qc = tc.get('query_curriculum', {})
        self.query_curriculum_enabled = bool(qc.get('enabled', False))
        self.query_curriculum_start_epoch = int(qc.get('start_epoch', 0))
        self.query_curriculum_end_epoch = int(qc.get('end_epoch', 0))

        # Build components
        self._build_dcff()
        self._build_model()
        self._build_datasets()
        self._build_optimizer()
        self._sync_map_finetune_state(0, force=True)

        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tb'))

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_trans = float('inf')
        self.best_metric_name = str(tc.get('best_metric', 'trans_median'))
        self.best_metric_value = float('-inf') if self.best_metric_name.startswith(('joint_', 'val_joint_', 'pct_', 'val_pct_')) else float('inf')
        self.best_metric_label = 'val_trans_median'
        self.latest_val_metrics: Dict[str, float] = {}
        self.epochs_since_best = 0
        tc = self.config.get('training', {})
        self.early_stop_patience = tc.get('early_stop_patience', 0)  # 0 = disabled

        # Resume / warmstart
        if resume_path:
            self._load_checkpoint(resume_path)
        elif warmstart_path:
            self._warmstart(warmstart_path)
        self._sync_map_finetune_state(self.epoch, force=True)

        if not self.save_epoch_checkpoints:
            self._cleanup_epoch_checkpoints()

    # ── DCFF Loading ──────────────────────────────────────────────────────

    def _build_dcff(self):
        """Load pre-trained DCFF model (frozen) for rendering."""
        torch.cuda.set_device(self.gpu)
        runtime = build_dcff_runtime(
            self.config,
            self.device,
            printer=self.logger.info,
        )
        self.gaussians = runtime.gaussians
        self.hash_grid = runtime.hash_grid
        self.dcff_renderer = runtime.renderer
        self.feat_sharp_fine = runtime.refiner
        self.feat_select = runtime.feat_select
        self.finetune_decoder = runtime.finetune_decoder
        self.finetune_fsm = runtime.finetune_fsm
        self.render_h = runtime.render_height
        self.render_w = runtime.render_width
        return

        cfg_dcff = self.config.get('dcff', {})
        dcff_ckpt_path = cfg_dcff.get('checkpoint')
        joint_ckpt_path = cfg_dcff.get('joint_checkpoint')
        ply_path = cfg_dcff.get('ply_path')

        if not dcff_ckpt_path or not os.path.isfile(dcff_ckpt_path):
            raise FileNotFoundError(
                f"DCFF checkpoint not found: {dcff_ckpt_path}\n"
                f"Set dcff.checkpoint in config or use --dcff_checkpoint"
            )
        if not ply_path or not os.path.isfile(ply_path):
            raise FileNotFoundError(
                f"PLY file not found: {ply_path}\n"
                f"Set dcff.ply_path in config"
            )

        torch.cuda.set_device(self.gpu)

        latent_dim = cfg_dcff.get('latent_dim', 32)
        feature_dim = cfg_dcff.get('feature_dim', 64)

        # 1. Load 2DGS geometry + latent
        self.gaussians = HybridGaussianModel(sh_degree=3, latent_dim=latent_dim)
        self.gaussians.load_ply(ply_path, freeze_geometry=True)
        self.gaussians.active_sh_degree = 3
        self.logger.info(f'Loaded {self.gaussians.num_points:,} Gaussians from {ply_path}')

        # Scene extent for hash grid
        xyz = self.gaussians.get_xyz.detach()
        scene_extent = float((xyz.max(dim=0).values - xyz.min(dim=0).values).max()) * 0.6
        self.logger.info(f'Scene extent: {scene_extent:.2f}')

        # Load DCFF training config if available
        dcff_config_path = os.path.join(os.path.dirname(dcff_ckpt_path), '..', 'config.yaml')
        dcff_cfg = {}
        if os.path.isfile(dcff_config_path):
            with open(dcff_config_path) as f:
                dcff_cfg = yaml.safe_load(f) or {}
            self.logger.info(f'Loaded DCFF config from {dcff_config_path}')

        hcfg = dcff_cfg.get('hash_grid', {})
        fcfg = dcff_cfg.get('fine_decoder', {})

        # 2. Build hash grid
        input_mode = hcfg.get('input_mode', 'implicit_scale')
        self.hash_grid = SpatialHashGrid(
            scene_extent=hcfg.get('scene_extent', scene_extent),
            feature_dim=feature_dim,
            input_mode=input_mode,
            latent_dim=latent_dim,
            n_levels=hcfg.get('n_levels', 16),
            n_features_per_level=hcfg.get('n_features_per_level', 2),
            log2_hashmap_size=hcfg.get('log2_hashmap_size', 19),
            base_resolution=hcfg.get('base_resolution', 16),
            max_resolution=hcfg.get('max_resolution', 2048),
            mlp_hidden=hcfg.get('mlp_hidden', 128),
            mlp_layers=hcfg.get('mlp_layers', 2),
            scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
            include_raw_scale=hcfg.get('include_raw_scale', True),
        ).to(self.device)

        # 3. Build renderer (wraps fine decoder)
        self.dcff_renderer = DeferredCascadedRenderer(
            hash_grid=self.hash_grid,
            latent_dim=latent_dim,
            fine_feature_dim=feature_dim,
            coarse_feature_dim=feature_dim,
            fine_hidden_dim=fcfg.get('hidden_dim', 128),
            fine_num_layers=fcfg.get('num_layers', 3),
            fine_use_viewdirs=fcfg.get('use_viewdirs', False),
            fine_view_degree=fcfg.get('view_degree', 2),
            fine_decoder_type=fcfg.get('type', 'mlp'),
            coarse_smoothing_kernel=cfg_dcff.get('coarse_smoothing_kernel', 1),
        ).to(self.device)

        # 4. Feature refinement module (auto-detect from DCFF config)
        refiner_type = dcff_cfg.get('refiner', {}).get('type', 'featsharp')
        if refiner_type == 'depth_guided':
            refiner_hidden = dcff_cfg.get('refiner', {}).get('hidden_dim', 128)
            self.feat_sharp_fine = DepthGuidedRefiner(
                feature_dim, hidden_dim=refiner_hidden).to(self.device)
            self.logger.info(f'  Using DepthGuidedRefiner (hidden={refiner_hidden})')
        else:
            self.feat_sharp_fine = FeatSharp(feature_dim).to(self.device)
            self.logger.info('  Using FeatSharp')

        # 5. Load checkpoint weights
        self.logger.info(f'Loading DCFF checkpoint: {dcff_ckpt_path}')
        ckpt = torch.load(dcff_ckpt_path, map_location=self.device)
        self.hash_grid.load_state_dict(ckpt['hash_grid_state'])
        self.dcff_renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
        if 'feat_sharp_fine_state' in ckpt:
            try:
                self.feat_sharp_fine.load_state_dict(
                    ckpt['feat_sharp_fine_state'], strict=True)
            except RuntimeError:
                self.logger.warning(
                    '  feat_sharp_fine shape mismatch — loading partial')
                self.feat_sharp_fine.load_state_dict(
                    ckpt['feat_sharp_fine_state'], strict=False)
        if 'latent' in ckpt:
            saved_latent = ckpt['latent'].to(self.device)
            if saved_latent.shape == self.gaussians._latent.shape:
                with torch.no_grad():
                    self.gaussians._latent.data.copy_(saved_latent)
                self.logger.info('  Restored latent embeddings from checkpoint')
            else:
                self.logger.warning(
                    f'  Latent shape mismatch: ckpt={saved_latent.shape} vs '
                    f'model={self.gaussians._latent.shape}, skipping'
                )

        if joint_ckpt_path:
            if not os.path.isfile(joint_ckpt_path):
                raise FileNotFoundError(f"Joint checkpoint not found: {joint_ckpt_path}")
            self.logger.info(f'Loading joint feature checkpoint: {joint_ckpt_path}')
            joint_ckpt = torch.load(joint_ckpt_path, map_location=self.device)
            joint_map_state = joint_ckpt.get('map_renderer_state_dict') or {}

            # Allow config to restrict which components are overridden
            allowed = cfg_dcff.get('joint_override_components', None)
            if allowed is not None:
                allowed = set(allowed)
                self.logger.info(f'  joint_override_components filter: {sorted(allowed)}')

            loaded_components = []
            if 'fine_decoder' in joint_map_state and (allowed is None or 'fine_decoder' in allowed):
                try:
                    self.dcff_renderer.fine_decoder.load_state_dict(joint_map_state['fine_decoder'])
                    loaded_components.append('fine_decoder')
                except (RuntimeError, KeyError) as e:
                    self.logger.warning(f'  Skipping joint fine_decoder (architecture changed): {e}')
            if 'feat_sharp' in joint_map_state and (allowed is None or 'feat_sharp' in allowed):
                try:
                    self.feat_sharp_fine.load_state_dict(joint_map_state['feat_sharp'])
                    loaded_components.append('feat_sharp')
                except (RuntimeError, KeyError) as e:
                    self.logger.warning(f'  Skipping joint feat_sharp (architecture changed): {e}')
            if 'hash_grid_mlp' in joint_map_state and (allowed is None or 'hash_grid_mlp' in allowed):
                try:
                    self.dcff_renderer.hash_grid.mlp.load_state_dict(joint_map_state['hash_grid_mlp'])
                    loaded_components.append('hash_grid_mlp')
                except (RuntimeError, KeyError) as e:
                    self.logger.warning(f'  Skipping joint hash_grid_mlp (architecture changed): {e}')
            if loaded_components:
                self.logger.info(
                    '  Overrode DCFF modules from joint checkpoint: %s',
                    ', '.join(loaded_components),
                )
            else:
                self.logger.warning(
                    '  joint checkpoint has no map_renderer_state_dict overrides; keeping base DCFF weights'
                )

        dcff_iter = ckpt.get('iteration', '?')
        self.logger.info(f'  DCFF loaded (iter {dcff_iter})')

        # 6. Freeze DCFF parameters (optionally unfreeze fine_decoder)
        self.gaussians._latent.requires_grad_(False)
        for p in self.hash_grid.parameters():
            p.requires_grad_(False)
        for p in self.dcff_renderer.parameters():
            p.requires_grad_(False)
        for p in self.feat_sharp_fine.parameters():
            p.requires_grad_(False)
        self.hash_grid.eval()
        self.dcff_renderer.eval()
        self.feat_sharp_fine.eval()

        # End-to-end decoder fine-tuning: unfreeze fine_decoder + feat_sharp
        self.finetune_decoder = cfg_dcff.get('finetune_decoder', False)
        if self.finetune_decoder:
            for p in self.dcff_renderer.fine_decoder.parameters():
                p.requires_grad_(True)
            for p in self.feat_sharp_fine.parameters():
                p.requires_grad_(True)
            self.dcff_renderer.fine_decoder.train()
            self.feat_sharp_fine.train()
            n_dec = sum(p.numel() for p in self.dcff_renderer.fine_decoder.parameters())
            n_fs = sum(p.numel() for p in self.feat_sharp_fine.parameters())
            self.logger.info(f'  Decoder fine-tuning enabled: {n_dec + n_fs:,} params unfrozen')

        # Rendering resolution
        self.render_w = cfg_dcff.get('render_width', 120)
        self.render_h = cfg_dcff.get('render_height', 68)
        self.logger.info(f'  Render resolution: {self.render_w}×{self.render_h}')

    # ── Model ─────────────────────────────────────────────────────────────

    def _build_model(self):
        """Build ConcatPoseNet (trainable)."""
        mcfg = self.config.get('model', {})
        tc = self.config.get('training', {})
        self.use_coarse = mcfg.get('use_coarse', False)
        self.use_gru = mcfg.get('use_gru', False)
        self.model = build_concat_pose_model(mcfg, self.device)
        local_matcher_init = mcfg.get('local_matcher_init_checkpoint')
        if local_matcher_init:
            load_local_matcher_weights(self.model, local_matcher_init, self.device)
            self.logger.info(f'  Loaded local matcher weights from {local_matcher_init}')
        local_flow_head_init = mcfg.get('local_flow_head_init_checkpoint')
        if local_flow_head_init:
            load_local_flow_head_weights(self.model, local_flow_head_init, self.device)
            self.logger.info(f'  Loaded local flow head weights from {local_flow_head_init}')
        if bool(tc.get('freeze_pose_model', False)):
            for param in self.model.parameters():
                param.requires_grad_(False)
            if bool(tc.get('train_feature_projection', False)):
                for module_name in ('proj_shared', 'proj_query', 'proj_render', 'cross_attn'):
                    module = getattr(self.model, module_name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad_(True)
        self.use_two_stage_refine = bool(getattr(self.model, 'use_two_stage_refine', False))

        n_params = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        mode_str = 'GRU' if self.use_gru else 'Concat'
        stage_str = 'two-stage' if self.use_two_stage_refine else 'single-stage'
        self.logger.info(f'ConcatPoseNet ({mode_str}, {stage_str}): {n_params:,} params ({n_train:,} trainable)')

    # ── Datasets ──────────────────────────────────────────────────────────

    def _build_datasets(self):
        """Build train/val datasets + DataLoaders."""
        dcfg = self.config.get('dataset', {})

        fine_hw = tuple(dcfg.get('fine_hw', [self.render_h, self.render_w]))
        coarse_hw = tuple(dcfg.get('coarse_hw', fine_hw))
        common_kwargs = dict(
            feature_dir=dcfg['feature_dir'],
            colmap_dir=dcfg['colmap_dir'],
            source_dir=dcfg.get('source_dir'),
            coarse_hw=coarse_hw,
            fine_hw=fine_hw,
            cache_in_memory=dcfg.get('cache_in_memory', True),
            normalize_features=dcfg.get('normalize_features', False),
        )
        teacher_feature_dir = dcfg.get('teacher_feature_dir') if self.query_curriculum_enabled else None
        source_dir = dcfg.get('source_dir')
        retrieval_train_split = dcfg.get('retrieval_train_split', dcfg.get('train_split'))

        def _validate_init_cache(config_key: str) -> Optional[str]:
            cache_path = dcfg.get(config_key)
            if cache_path in (None, ''):
                return None
            resolved = resolve_repo_path(cache_path, must_exist=True)
            assert resolved is not None
            return str(resolved)

        train_init_poses_path = _validate_init_cache('train_init_poses_path')
        val_init_poses_path = _validate_init_cache('val_init_poses_path')
        self.train_uses_explicit_init = train_init_poses_path is not None
        self.val_uses_explicit_init = val_init_poses_path is not None
        self.train_jitter_loaded_init = bool(dcfg.get('train_jitter_loaded_init', False))

        def _build_dataset(
            *,
            split: str,
            split_file: Optional[str],
            noise_rot_deg: float,
            noise_trans_m: float,
            teacher_feature_dir_override: Optional[str] = None,
            init_poses_path: Optional[str] = None,
        ):
            dataset_kwargs = dict(
                split=split,
                split_file=split_file,
                noise_rot_deg=noise_rot_deg,
                noise_trans_m=noise_trans_m,
                teacher_feature_dir=teacher_feature_dir_override,
                **common_kwargs,
            )
            if init_poses_path is None:
                return RadioLocDataset(**dataset_kwargs)
            return RadioLocRetrievalDataset(
                init_poses_path=init_poses_path,
                retrieval_train_split=retrieval_train_split,
                jitter_loaded_init=bool(dcfg.get('train_jitter_loaded_init', False)) if split == 'train' else False,
                sample_topk_init=bool(dcfg.get('train_sample_topk_init', False)) if split == 'train' else False,
                sample_topk_prob=float(dcfg.get('train_sample_topk_prob', 0.0)) if split == 'train' else 0.0,
                **dataset_kwargs,
            )

        self.train_dataset = _build_dataset(
            split='train',
            split_file=dcfg.get('train_split'),
            noise_rot_deg=self.noise_rot_start,
            noise_trans_m=self.noise_trans_start,
            teacher_feature_dir_override=teacher_feature_dir,
            init_poses_path=train_init_poses_path,
        )
        # Set per-sample noise range if configured.
        if not self.train_uses_explicit_init and self.noise_rot_min is not None:
            self.train_dataset.noise_rot_min = self.noise_rot_min
            self.train_dataset.noise_trans_min = self.noise_trans_min
            self.logger.info(f'  Per-sample noise range: rot=[{self.noise_rot_min}°, {self.noise_rot_start}°] '
                             f'trans=[{self.noise_trans_min}m, {self.noise_trans_start}m]')
        if self.train_uses_explicit_init:
            self.logger.info(f'  Train init poses: {train_init_poses_path}')
            self.logger.info(f'  Train init stats: {getattr(self.train_dataset, "init_stats", {})}')
        self.val_dataset = _build_dataset(
            split='test',
            split_file=dcfg.get('test_split'),
            noise_rot_deg=self.val_noise_deg,
            noise_trans_m=self.val_noise_m,
            init_poses_path=val_init_poses_path,
        )
        if self.val_uses_explicit_init:
            self.logger.info(f'  Val init poses: {val_init_poses_path}')
            self.logger.info(f'  Val init stats: {getattr(self.val_dataset, "init_stats", {})}')

        # Store intrinsics (at COLMAP native resolution)
        self.intrinsics = self.train_dataset.intrinsics

        # Read original COLMAP camera resolution for proper scaling
        colmap_dir = dcfg['colmap_dir']
        colmap_cameras = read_colmap_cameras(
            os.path.join(colmap_dir, 'cameras.bin'))
        first_cam = next(iter(colmap_cameras.values()))
        self.orig_img_hw = (int(first_cam.height), int(first_cam.width))

        if self.intrinsics:
            self.logger.info(
                f'Intrinsics (native {self.orig_img_hw[1]}×{self.orig_img_hw[0]}): '
                f'fx={self.intrinsics["fx"]:.1f} '
                f'fy={self.intrinsics["fy"]:.1f} '
                f'cx={self.intrinsics["cx"]:.1f} '
                f'cy={self.intrinsics["cy"]:.1f}'
            )
            # Set on model for intrinsics scaling
            self.model.BASE_INTRINSICS = self.intrinsics
            self.model.IMG_HW = self.orig_img_hw

        tc = self.config.get('training', {})
        batch_size = tc.get('batch_size', 6)
        num_workers = int(tc.get('num_workers', 4))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=min(batch_size, 6),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.logger.info(f'Train: {len(self.train_dataset)} samples, '
                         f'{len(self.train_loader)} batches')
        self.logger.info(f'Val: {len(self.val_dataset)} samples, '
                         f'{len(self.val_loader)} batches')
        self.logger.info(f'DataLoader workers: {num_workers}')

    # ── Optimizer ─────────────────────────────────────────────────────────

    def _build_optimizer(self):
        """AdamW + CosineAnnealingLR."""
        tc = self.config.get('training', {})
        lr = tc.get('lr', 3e-4)
        weight_decay = tc.get('weight_decay', 1e-5)

        param_groups = []
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        if model_params:
            param_groups.append({'params': model_params, 'lr': lr})

        # End-to-end decoder fine-tuning at lower LR
        if getattr(self, 'finetune_decoder', False):
            decoder_lr = tc.get('decoder_lr', lr * 0.1)
            decoder_params = list(self.dcff_renderer.fine_decoder.parameters()) + list(self.feat_sharp_fine.parameters())
            coarse_fusion = getattr(self.dcff_renderer, 'coarse_carrier_fusion', None)
            if coarse_fusion is not None:
                decoder_params += list(coarse_fusion.parameters())
            param_groups.append({
                'params': decoder_params,
                'lr': decoder_lr,
            })
            self.logger.info(f'  Decoder LR: {decoder_lr:.1e} (main: {lr:.1e})')
        if getattr(self, 'finetune_fsm', False) and self.feat_select is not None:
            fsm_lr = tc.get('fsm_lr', tc.get('decoder_lr', lr * 0.1))
            param_groups.append({
                'params': list(self.feat_select.parameters()),
                'lr': fsm_lr,
            })
            self.logger.info(f'  FSM LR: {fsm_lr:.1e} (main: {lr:.1e})')

        self.optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(tc.get('epochs', 200), 1),
            eta_min=tc.get('min_lr', 1e-6),
        )

    def _iter_decoder_finetune_modules(self):
        yield self.dcff_renderer.fine_decoder
        yield self.feat_sharp_fine
        coarse_fusion = getattr(self.dcff_renderer, 'coarse_carrier_fusion', None)
        if coarse_fusion is not None:
            yield coarse_fusion

    @staticmethod
    def _set_module_trainable(module: nn.Module | None, enabled: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _sync_map_finetune_state(self, epoch: int, force: bool = False) -> None:
        decoder_active = bool(
            getattr(self, 'finetune_decoder', False)
            and epoch >= self.map_finetune_start_epoch
        )
        fsm_active = bool(
            getattr(self, 'finetune_fsm', False)
            and self.feat_select is not None
            and epoch >= self.fsm_finetune_start_epoch
        )

        if force or decoder_active != self._map_decoder_active:
            for module in self._iter_decoder_finetune_modules():
                self._set_module_trainable(module, decoder_active)
            if decoder_active:
                self.logger.info(
                    f'  Map decoder fine-tuning active from epoch {epoch} '
                    f'(start={self.map_finetune_start_epoch})')
            elif getattr(self, 'finetune_decoder', False):
                self.logger.info(
                    f'  Map decoder frozen until epoch {self.map_finetune_start_epoch}')
            self._map_decoder_active = decoder_active

        if force or fsm_active != self._map_fsm_active:
            self._set_module_trainable(self.feat_select, fsm_active)
            if fsm_active:
                self.logger.info(
                    f'  FSM fine-tuning active from epoch {epoch} '
                    f'(start={self.fsm_finetune_start_epoch})')
            elif getattr(self, 'finetune_fsm', False) and self.feat_select is not None:
                self.logger.info(
                    f'  FSM frozen until epoch {self.fsm_finetune_start_epoch}')
            self._map_fsm_active = fsm_active

    def _set_map_train_mode(self, enabled: bool) -> None:
        decoder_train = bool(enabled and self._map_decoder_active)
        fsm_train = bool(enabled and self._map_fsm_active)
        self.dcff_renderer.train(decoder_train or fsm_train)
        self.dcff_renderer.fine_decoder.train(decoder_train)
        self.feat_sharp_fine.train(decoder_train)
        coarse_fusion = getattr(self.dcff_renderer, 'coarse_carrier_fusion', None)
        if coarse_fusion is not None:
            coarse_fusion.train(decoder_train)
        if self.feat_select is not None:
            self.feat_select.train(fsm_train)

    # ── DCFF Rendering ────────────────────────────────────────────────────

    def _intrinsics_to_K(self, intrinsics: Dict[str, float]) -> torch.Tensor:
        """Convert intrinsics dict to 3×3 K matrix on device."""
        return intrinsics_to_K(intrinsics, self.device)

    @torch.no_grad()
    def _render_dcff_at_pose(
        self,
        pose_w2c: torch.Tensor,
        render_coarse: bool | None = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Render DCFF fine/coarse features + depth at a given w2c pose.

        Args:
            pose_w2c: (4, 4) world-to-camera transform

        Returns:
            dict containing fine/coarse features, depth, and FSM aux outputs
        """
        render_intr = self.model._scale_intrinsics(self.render_h, self.render_w)
        K = self._intrinsics_to_K(render_intr)
        render_bundle = render_feature_bundle_batch(
            self.gaussians,
            self.dcff_renderer,
            self.feat_sharp_fine,
            pose_w2c.unsqueeze(0).to(self.device),
            K,
            self.render_h,
            self.render_w,
            render_coarse=render_coarse,
        )
        return render_bundle

    def _render_dcff_at_pose_differentiable(
        self,
        pose_w2c: torch.Tensor,
        render_coarse: bool | None = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Render DCFF with gradient through fine_decoder + feat_sharp."""
        viewmat = pose_w2c.float().to(self.device)
        render_intr = self.model._scale_intrinsics(self.render_h, self.render_w)
        K = self._intrinsics_to_K(render_intr)
        use_coarse_for_fsm = bool(getattr(self.dcff_renderer, '_fsm_use_coarse', False))
        should_render_coarse = use_coarse_for_fsm if render_coarse is None else bool(render_coarse)

        # Rasterize z_map without gradient (geometry is frozen)
        with torch.no_grad():
            result = self.dcff_renderer(
                self.gaussians,
                viewmat=viewmat,
                K=K,
                width=self.render_w,
                height=self.render_h,
                render_coarse=should_render_coarse,
                feature_height=self.render_h,
                feature_width=self.render_w,
            )
            z_map = result['z_map'].detach()
            depth = result['depth'].detach()
            alpha = result.get('alpha')
            scale_map = result.get('scale_map')
            if alpha is not None:
                alpha = alpha.detach()
            if scale_map is not None:
                scale_map = scale_map.detach()

        if getattr(self.dcff_renderer, 'split_latent', False):
            fine_dim = int(self.dcff_renderer.fine_latent_dim)
            coarse_dim = int(self.dcff_renderer.coarse_latent_dim)
            z_fine_map = z_map[:, :fine_dim]
            z_coarse_map = z_map[:, fine_dim:fine_dim + coarse_dim]
        else:
            z_fine_map = z_map
            z_coarse_map = z_map

        position_map = None
        if self.dcff_renderer.fine_decoder.use_viewdirs or should_render_coarse:
            position_map = self.dcff_renderer.depth_to_position_map(depth, K, viewmat)

        coarse_features = None
        if should_render_coarse:
            if self.dcff_renderer.coarse_mode in {'spatial_direct', 'spatial_full_direct'}:
                coarse_input_map = (
                    z_map
                    if getattr(self.dcff_renderer, 'coarse_direct_uses_full_latent', False)
                    else z_coarse_map
                )
                coarse_features, _, _ = self.dcff_renderer.coarse_carrier_fusion(
                    coarse_input_map,
                    None,
                )
            else:
                coarse_features = self.dcff_renderer.decode_coarse(
                    position_map=position_map,
                    alpha=alpha if alpha is not None else torch.ones_like(depth),
                    z_map=z_coarse_map,
                    scale_map=scale_map,
                    viewmat=viewmat,
                ).float()
                if self.dcff_renderer.coarse_carrier_fusion is not None:
                    coarse_features, _, _ = self.dcff_renderer.coarse_carrier_fusion(
                        z_coarse_map,
                        coarse_features,
                    )

        # Re-decode through fine_decoder WITH gradient
        fine_feat = self.dcff_renderer.decode_fine(
            z_fine_map,
            position_map=position_map,
            viewmat=viewmat,
        ).float()

        post_result = {
            'fine_features': fine_feat,
            'coarse_features': coarse_features,
            'depth': depth,
            'alpha': alpha if alpha is not None else torch.ones_like(depth),
        }
        post_result = _apply_dcff_postprocess(
            post_result,
            self.render_h,
            self.render_w,
            feat_sharp=self.feat_sharp_fine,
            feat_select=self.feat_select,
            use_coarse_for_fsm=use_coarse_for_fsm,
            temperature=0.5,
            hard=False,
        )

        return {
            'fine_features': post_result['fine_features'].float(),
            'coarse_features': post_result.get('coarse_features').float() if post_result.get('coarse_features') is not None else None,
            'depth': depth,
            'alpha': post_result.get('alpha').float() if post_result.get('alpha') is not None else torch.ones_like(depth),
            'fsm_spatial_conf': post_result.get('fsm_spatial_conf').float() if post_result.get('fsm_spatial_conf') is not None else None,
            'fsm_channel_weights': post_result.get('fsm_channel_weights').float() if post_result.get('fsm_channel_weights') is not None else None,
        }

    @torch.no_grad()
    def _render_batch(
        self,
        poses_w2c: torch.Tensor,
        render_coarse: bool | None = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Render DCFF features + depth for a batch of poses.

        Args:
            poses_w2c: (B, 4, 4)

        Returns:
            dict containing batched fine/coarse features, depth, and FSM aux outputs
        """
        render_intr = self.model._scale_intrinsics(self.render_h, self.render_w)
        K = self._intrinsics_to_K(render_intr)
        return render_feature_bundle_batch(
            self.gaussians,
            self.dcff_renderer,
            self.feat_sharp_fine,
            poses_w2c.to(self.device),
            K,
            self.render_h,
            self.render_w,
            render_coarse=render_coarse,
        )

    def _render_batch_differentiable(
        self,
        poses_w2c: torch.Tensor,
        render_coarse: bool | None = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Render with gradient through fine_decoder for end-to-end training."""
        B = poses_w2c.shape[0]
        fine_list, depth_list = [], []
        alpha_list = []
        coarse_list = []
        fsm_spatial_list = []
        fsm_channel_list = []

        for i in range(B):
            rendered_i = self._render_dcff_at_pose_differentiable(poses_w2c[i], render_coarse=render_coarse)
            fine_list.append(rendered_i['fine_features'].squeeze(0))
            depth_list.append(rendered_i['depth'].squeeze(0).squeeze(0))
            alpha_list.append(rendered_i['alpha'].squeeze(0))
            if rendered_i['coarse_features'] is not None:
                coarse_list.append(rendered_i['coarse_features'].squeeze(0))
            if rendered_i['fsm_spatial_conf'] is not None:
                fsm_spatial_list.append(rendered_i['fsm_spatial_conf'].squeeze(0))
            if rendered_i['fsm_channel_weights'] is not None:
                fsm_channel_list.append(rendered_i['fsm_channel_weights'].squeeze(0))

        render_bundle: Dict[str, Optional[torch.Tensor]] = {
            'fine_features': torch.stack(fine_list, dim=0),
            'depth': torch.stack(depth_list, dim=0),
            'alpha': torch.stack(alpha_list, dim=0),
            'coarse_features': torch.stack(coarse_list, dim=0) if len(coarse_list) == B else None,
            'fsm_spatial_conf': torch.stack(fsm_spatial_list, dim=0) if len(fsm_spatial_list) == B else None,
            'fsm_channel_weights': torch.stack(fsm_channel_list, dim=0) if len(fsm_channel_list) == B else None,
        }

        return render_bundle

    def _render_bundle_batch(
        self,
        poses_w2c: torch.Tensor,
        *,
        differentiable: bool = False,
        render_coarse: bool | None = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if differentiable:
            return self._render_batch_differentiable(poses_w2c, render_coarse=render_coarse)
        return self._render_batch(poses_w2c, render_coarse=render_coarse)

    # ── Noise Curriculum ──────────────────────────────────────────────────

    def _update_noise_for_epoch(self, epoch: int):
        """Linearly ramp noise from start to end over warmup epochs."""
        if getattr(self, 'train_uses_explicit_init', False):
            if not getattr(self, 'train_jitter_loaded_init', False):
                if epoch == 0:
                    self.logger.info('  Noise curriculum disabled: explicit train init poses are in use')
                return

        t = min(epoch / max(self.noise_warmup_epochs, 1), 1.0)
        rot_deg = self.noise_rot_start + (self.noise_rot_end - self.noise_rot_start) * t
        trans_m = self.noise_trans_start + (self.noise_trans_end - self.noise_trans_start) * t

        self.train_dataset.noise_rot_deg = rot_deg
        self.train_dataset.noise_trans_m = trans_m

        if epoch % 10 == 0 or epoch == 0:
            self.logger.info(f'  Noise curriculum E{epoch}: '
                             f'rot={rot_deg:.1f}° trans={trans_m:.3f}m')

    def _student_query_alpha(self) -> float:
        if not self.query_curriculum_enabled:
            return 1.0

        start = self.query_curriculum_start_epoch
        end = self.query_curriculum_end_epoch
        if end <= start:
            return 1.0 if self.epoch >= end else 0.0
        if self.epoch <= start:
            return 0.0
        if self.epoch >= end:
            return 1.0
        return float(self.epoch - start) / float(end - start)

    def _apply_query_curriculum(
        self,
        batch: Dict,
        query_fine: torch.Tensor,
        query_coarse: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        if not self.query_curriculum_enabled:
            return query_fine, query_coarse, None

        teacher_query_fine = batch.get('teacher_query_fine')
        if teacher_query_fine is None:
            return query_fine, query_coarse, None

        alpha = self._student_query_alpha()
        teacher_query_fine = teacher_query_fine.to(self.device)
        if alpha <= 0.0:
            query_fine = teacher_query_fine
        elif alpha < 1.0:
            query_fine = torch.lerp(teacher_query_fine, query_fine, alpha)

        teacher_query_coarse = batch.get('teacher_query_coarse')
        if query_coarse is not None and teacher_query_coarse is not None and self.use_coarse:
            teacher_query_coarse = teacher_query_coarse.to(self.device)
            if alpha <= 0.0:
                query_coarse = teacher_query_coarse
            elif alpha < 1.0:
                query_coarse = torch.lerp(teacher_query_coarse, query_coarse, alpha)

        return query_fine, query_coarse, alpha

    def _pose_perturb_rank_loss(
        self,
        query_fine: torch.Tensor,
        pose_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Rank GT-pose renders ahead of perturbed-pose renders."""
        B = pose_gt.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            noise = torch.randn(B, 6, device=self.device, dtype=torch.float32)
            noise[:, :3] *= self.pose_rank_trans_m
            noise[:, 3:] *= math.radians(self.pose_rank_rot_deg)
            pose_neg = torch.bmm(se3_exp(noise), pose_gt.float())

        pos_feat = self._render_batch_differentiable(
            pose_gt,
            render_coarse=False,
        )['fine_features']
        neg_feat = self._render_batch_differentiable(
            pose_neg,
            render_coarse=False,
        )['fine_features']

        pos_dist = feature_cosine_distance_per_sample(query_fine, pos_feat)
        neg_dist = feature_cosine_distance_per_sample(query_fine, neg_feat)
        rank_loss = F.relu(self.pose_rank_margin + pos_dist - neg_dist).mean()
        return rank_loss, {
            'pose_rank_loss': rank_loss.item(),
            'pose_rank_pos_dist': pos_dist.mean().item(),
            'pose_rank_neg_dist': neg_dist.mean().item(),
        }

    def _sample_loc_rank_offsets(
        self,
        batch_size: int,
        negative_index: int,
    ) -> torch.Tensor:
        distances = torch.tensor(
            self.loc_rank_distances_m or [0.05],
            device=self.device,
            dtype=torch.float32,
        )
        dist_idx = torch.randint(0, distances.numel(), (batch_size,), device=self.device)
        axes = (torch.arange(batch_size, device=self.device) + int(negative_index)) % 3
        signs = torch.where(
            torch.rand(batch_size, device=self.device) < 0.5,
            -torch.ones(batch_size, device=self.device),
            torch.ones(batch_size, device=self.device),
        )
        offsets = torch.zeros(batch_size, 3, device=self.device, dtype=torch.float32)
        offsets[torch.arange(batch_size, device=self.device), axes] = signs * distances[dist_idx]
        return offsets

    @staticmethod
    def _depth_alpha_mask(bundle: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        depth = bundle['depth']
        mask = (depth.float() > 0.05).unsqueeze(1) if depth.ndim == 3 else (depth.float() > 0.05)
        alpha = bundle.get('alpha')
        if alpha is not None:
            alpha_f = alpha.float()
            if alpha_f.ndim == 3:
                alpha_f = alpha_f.unsqueeze(1)
            mask = mask.float() * (alpha_f > 0.1).float()
        return mask.float()

    def _render_localization_train_bundle(
        self,
        poses_w2c: torch.Tensor,
        *,
        render_coarse: bool | None = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self._render_bundle_batch(
            poses_w2c,
            differentiable=bool(self._map_decoder_active or self._map_fsm_active),
            render_coarse=render_coarse,
        )

    def _localization_feature_pair(
        self,
        query_fine: torch.Tensor,
        rendered_fine: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query_fine.float()
        rendered = rendered_fine.float()
        if query.shape[-2:] != rendered.shape[-2:]:
            query = F.interpolate(query, rendered.shape[-2:], mode='bilinear', align_corners=False)
        if not self.loc_use_projection:
            return query, rendered
        return self.model._project_for_local_corr(query, rendered)

    def _loc_cm_perturb_rank_loss(
        self,
        query_fine: torch.Tensor,
        pose_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Rank GT-pose DCFF features ahead of centimetre-scale pose negatives."""
        B = pose_gt.shape[0]
        query = query_fine.detach() if self.loc_detach_query else query_fine
        pos_bundle = self._render_localization_train_bundle(pose_gt, render_coarse=False)
        pos_feat = pos_bundle['fine_features']
        pos_mask = self._depth_alpha_mask(pos_bundle)
        q_pos, r_pos = self._localization_feature_pair(query, pos_feat)
        pos_dist = masked_feature_cosine_distance_per_sample(q_pos, r_pos, pos_mask)

        losses = []
        neg_dists = []
        margins = []
        neg_mm = []
        for k in range(self.loc_rank_negatives):
            offsets = self._sample_loc_rank_offsets(B, k)
            pose_neg = perturb_w2c_camera_center(
                pose_gt,
                offsets,
                frame=self.loc_rank_frame,
            )
            neg_bundle = self._render_localization_train_bundle(pose_neg, render_coarse=False)
            neg_mask = pos_mask * self._depth_alpha_mask(neg_bundle)
            q_neg, r_neg = self._localization_feature_pair(query, neg_bundle['fine_features'])
            neg_dist = masked_feature_cosine_distance_per_sample(q_neg, r_neg, neg_mask)
            margin = self.loc_rank_margin + self.loc_rank_margin_per_m * torch.linalg.norm(offsets, dim=1)
            losses.append(F.relu(margin + pos_dist - neg_dist))
            neg_dists.append(neg_dist)
            margins.append(margin)
            neg_mm.append(torch.linalg.norm(offsets, dim=1) * 1000.0)

        loss_stack = torch.stack(losses, dim=0)
        neg_dist_stack = torch.stack(neg_dists, dim=0)
        margin_stack = torch.stack(margins, dim=0)
        neg_mm_stack = torch.stack(neg_mm, dim=0)
        if self.loc_rank_hard_mining:
            rank_loss = loss_stack.max(dim=0).values.mean()
            hard_idx = neg_dist_stack.argmin(dim=0, keepdim=True)
            hard_neg_dist = neg_dist_stack.gather(0, hard_idx).squeeze(0)
            hard_margin = margin_stack.gather(0, hard_idx).squeeze(0)
            hard_neg_mm = neg_mm_stack.gather(0, hard_idx).squeeze(0)
        else:
            rank_loss = loss_stack.mean()
            hard_neg_dist = neg_dist_stack.mean(dim=0)
            hard_margin = margin_stack.mean(dim=0)
            hard_neg_mm = neg_mm_stack.mean(dim=0)

        with torch.no_grad():
            rank_acc = (hard_neg_dist > pos_dist + hard_margin).float().mean()
            gap = hard_neg_dist - pos_dist

        return rank_loss, {
            'loc_rank_loss': float(rank_loss.detach().item()),
            'loc_rank_pos_dist': float(pos_dist.detach().mean().item()),
            'loc_rank_neg_dist': float(hard_neg_dist.detach().mean().item()),
            'loc_rank_gap': float(gap.detach().mean().item()),
            'loc_rank_acc': float(rank_acc.detach().item()),
            'loc_rank_neg_mm': float(hard_neg_mm.detach().mean().item()),
        }

    def _feature_metric_training_loss(
        self,
        query_fine: torch.Tensor,
        pose_ref: torch.Tensor,
        pose_gt: torch.Tensor,
        intrinsics: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        query = query_fine.detach() if self.feature_metric_detach_query else query_fine
        bundle = self._render_localization_train_bundle(pose_ref, render_coarse=False)
        valid_mask = self._depth_alpha_mask(bundle)
        query_for_solve, rendered_for_solve = self._localization_feature_pair(
            query,
            bundle['fine_features'],
        )
        return feature_metric_pose_update_loss(
            query_for_solve,
            rendered_for_solve,
            bundle['depth'],
            pose_ref,
            pose_gt,
            intrinsics,
            damping=self.feature_metric_damping,
            valid_mask=valid_mask,
            normalize_features=self.feature_metric_normalize and not self.loc_use_projection,
            update_scale=self.feature_metric_update_scale,
            rot_weight=self.feature_metric_rot_weight,
            trans_weight=self.feature_metric_trans_weight,
        )

    # ── Training Step ─────────────────────────────────────────────────────

    def _train_step(
        self,
        batch: Dict,
        use_pose_loss: bool,
    ) -> Dict[str, float]:
        """Single training step with optional outer iteration refinement."""
        query_fine = batch['query_fine'].to(self.device)
        query_coarse = batch.get('query_coarse')
        if query_coarse is not None and self.use_coarse:
            query_coarse = query_coarse.to(self.device)
        else:
            query_coarse = None
        query_fine, query_coarse, student_query_alpha = self._apply_query_curriculum(
            batch, query_fine, query_coarse,
        )
        pose_gt = batch['pose_gt'].to(self.device)
        pose_start = batch['pose_init'].to(self.device)
        pose_cur = pose_start

        flow_hw = (self.render_h, self.render_w)
        render_intr = self.model._scale_intrinsics(self.render_h, self.render_w)

        # Phase1 is pure flow pretraining.  Do not feed an untrained WLS update
        # back into the second outer step; that makes the flow target depend on
        # the model's own early mistakes.  Iterative closed-loop training starts
        # once pose loss is active.
        N = self.outer_iters_train if use_pose_loss else 1
        all_metrics = {}
        total_loss_val = 0.0
        differentiable_render = self._map_decoder_active or self._map_fsm_active
        if student_query_alpha is not None:
            all_metrics['student_query_alpha'] = float(student_query_alpha)

        def render_batch_fn(poses_w2c: torch.Tensor, render_coarse: bool | None) -> Dict[str, Optional[torch.Tensor]]:
            return self._render_bundle_batch(
                poses_w2c,
                differentiable=differentiable_render,
                render_coarse=render_coarse,
            )

        if self.feature_only_train:
            all_metrics: Dict[str, float] = {'outer_iters_used': 0.0}
            total_loss_val = 0.0
            pose_start_bundle: Optional[Dict[str, Optional[torch.Tensor]]] = None
            localization_feature_train_active = has_trainable_localization_feature_path(
                self.model,
                loc_use_projection=self.loc_use_projection,
                map_decoder_active=self._map_decoder_active,
                map_fsm_active=self._map_fsm_active,
            )

            def get_pose_start_bundle() -> Dict[str, Optional[torch.Tensor]]:
                nonlocal pose_start_bundle
                if pose_start_bundle is None:
                    pose_start_bundle = self._render_localization_train_bundle(
                        pose_start,
                        render_coarse=False,
                    )
                return pose_start_bundle

            if (
                self.corr_flow_weight > 0
                or self.corr_ce_weight > 0
                or self.corr_subpixel_ce_weight > 0
            ):
                corr_bundle = get_pose_start_bundle()
                with torch.no_grad():
                    gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
                        pose_start,
                        pose_gt,
                        corr_bundle['depth'],
                        flow_hw,
                        render_intr,
                    )
                corr_total_loss = torch.tensor(0.0, device=self.device)
                if self.corr_ce_weight > 0:
                    corr_ce_loss, corr_ce_metrics = local_correlation_ce_loss(
                        self.model,
                        corr_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                        radius=int(getattr(self.model, 'local_radius', 4)),
                        temperature=self.corr_ce_temperature,
                    )
                    corr_total_loss = corr_total_loss + self.corr_ce_weight * corr_ce_loss
                    for k, v in corr_ce_metrics.items():
                        all_metrics[k] = v
                if self.corr_subpixel_ce_weight > 0:
                    corr_subpx_loss, corr_subpx_metrics = local_correlation_subpixel_ce_loss(
                        self.model,
                        corr_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                        radius=int(getattr(self.model, 'local_radius', 4)),
                        temperature=self.corr_subpixel_ce_temperature,
                    )
                    corr_total_loss = (
                        corr_total_loss
                        + self.corr_subpixel_ce_weight * corr_subpx_loss
                    )
                    for k, v in corr_subpx_metrics.items():
                        all_metrics[k] = v
                if self.corr_flow_weight > 0:
                    corr_flow_loss, corr_flow_metrics = local_correlation_soft_flow_loss(
                        self.model,
                        corr_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                        radius=int(getattr(self.model, 'local_radius', 4)),
                        temperature=self.corr_flow_temperature,
                    )
                    corr_total_loss = corr_total_loss + self.corr_flow_weight * corr_flow_loss
                    for k, v in corr_flow_metrics.items():
                        all_metrics[k] = v
                self.scaler.scale(corr_total_loss).backward()
                total_loss_val += corr_total_loss.item()

            if localization_feature_train_active and self.feat_match_weight > 0:
                ref_gt = self._render_localization_train_bundle(pose_gt, render_coarse=False)['fine_features']
                ref_norm = F.normalize(ref_gt.float(), dim=1)
                q_norm = F.normalize(query_fine.float(), dim=1)
                cos_sim = (ref_norm * q_norm).sum(dim=1).mean()
                feat_match_loss = 1.0 - cos_sim
                self.scaler.scale(self.feat_match_weight * feat_match_loss).backward()
                all_metrics['feat_match_loss'] = feat_match_loss.item()
                all_metrics['feat_cos_sim'] = cos_sim.item()
                total_loss_val += self.feat_match_weight * feat_match_loss.item()

            if (
                localization_feature_train_active
                and self.loc_rank_weight > 0
                and self.epoch >= self.loc_rank_start_epoch
            ):
                loc_rank_loss, loc_rank_metrics = self._loc_cm_perturb_rank_loss(
                    query_fine,
                    pose_gt,
                )
                self.scaler.scale(self.loc_rank_weight * loc_rank_loss).backward()
                for k, v in loc_rank_metrics.items():
                    all_metrics[k] = v
                total_loss_val += self.loc_rank_weight * loc_rank_loss.item()

            if (
                localization_feature_train_active
                and self.feature_metric_weight > 0
                and self.epoch >= self.feature_metric_start_epoch
            ):
                fm_loss, fm_metrics = self._feature_metric_training_loss(
                    query_fine,
                    pose_start,
                    pose_gt,
                    render_intr,
                )
                self.scaler.scale(self.feature_metric_weight * fm_loss).backward()
                for k, v in fm_metrics.items():
                    all_metrics[k] = v
                total_loss_val += self.feature_metric_weight * fm_loss.item()

            all_metrics['total_loss'] = total_loss_val
            if total_loss_val <= 0:
                all_metrics['nan_step'] = True
            return all_metrics

        for outer_i in range(N):
            iter_state = run_model_refine_iteration(
                self.model,
                render_batch_fn,
                query_fine,
                query_coarse,
                pose_cur,
                render_intr,
                outer_iter=outer_i,
                autocast_enabled=self.use_amp,
            )
            pred = iter_state['fine_pred']
            fine_bundle = iter_state['fine_bundle']
            depth = fine_bundle['depth']
            pose_mid = iter_state['pose_mid']
            coarse_pred = iter_state['coarse_pred']

            # 3. GT flow
            with torch.no_grad():
                gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
                    pose_mid, pose_gt, depth, flow_hw, render_intr,
                )
                flow_weight_map = None
                if self.flow_observability_weight > 0:
                    flow_weight_map = compute_observability_flow_weight(
                        depth,
                        flow_hw,
                        render_intr,
                        valid_mask=gt_valid,
                        mode=self.flow_observability_mode,
                        strength=self.flow_observability_weight,
                        max_weight=self.flow_observability_max_weight,
                    )

            # 4. Losses
            with autocast(enabled=self.use_amp):
                iter_loss = torch.tensor(0.0, device=self.device)
                coarse_metrics = {}

                if coarse_pred is not None and use_pose_loss:
                    coarse_loss, coarse_pose_metrics = pose_loss(
                        coarse_pred['delta_xi'], pose_cur, pose_gt,
                        rot_weight=self.rot_weight,
                        trans_weight=self.trans_weight,
                        rot_loss_type=self.rot_loss_type,
                        loss_mode=self.pose_loss_mode,
                    )
                    iter_loss = iter_loss + self.coarse_pose_weight * coarse_loss
                    coarse_metrics = {
                        f'coarse_{k}': v for k, v in coarse_pose_metrics.items()
                    }

                # Flow loss — RAFT-style sequence loss if GRU provides flow_preds
                if 'flow_preds' in pred and len(pred['flow_preds']) > 1:
                    n_preds = len(pred['flow_preds'])
                    gamma_seq = 0.8
                    seq_flow_loss = torch.tensor(0.0, device=self.device)
                    for pi, fp in enumerate(pred['flow_preds']):
                        w = gamma_seq ** (n_preds - 1 - pi)
                        fl, _ = flow_loss_fn(
                            fp,
                            gt_flow,
                            gt_valid,
                            weight_map=flow_weight_map,
                        )
                        seq_flow_loss = seq_flow_loss + w * fl
                    seq_flow_loss = seq_flow_loss / n_preds
                    iter_loss = iter_loss + self.flow_weight * seq_flow_loss
                    # EPE metric from final prediction
                    _, f_metrics = flow_loss_fn(
                        pred['flow'],
                        gt_flow,
                        gt_valid,
                        weight_map=flow_weight_map,
                    )
                    f_metrics['flow_loss'] = seq_flow_loss.item()
                else:
                    f_loss, f_metrics = flow_loss_fn(
                        pred['flow'],
                        gt_flow,
                        gt_valid,
                        weight_map=flow_weight_map,
                    )
                    iter_loss = iter_loss + self.flow_weight * f_loss

                if self.init_flow_weight > 0 and 'init_flow' in pred:
                    init_f_loss, init_f_metrics = flow_loss_fn(
                        pred['init_flow'],
                        gt_flow,
                        gt_valid,
                        weight_map=flow_weight_map,
                    )
                    iter_loss = iter_loss + self.init_flow_weight * init_f_loss
                    f_metrics.update({
                        'init_flow_loss': init_f_loss.item(),
                        'init_flow_epe': init_f_metrics.get('flow_epe', 0.0),
                    })

                # Confidence regularization
                c_loss, c_metrics = confidence_regularization_loss(
                    pred['confidence'],
                )
                iter_loss = iter_loss + self.conf_reg_weight * c_loss

                # Confidence NLL (self-supervised: high conf where flow is good)
                if self.conf_nll_weight > 0:
                    cnll_loss, cnll_metrics = confidence_nll_loss(
                        pred['flow'], gt_flow, pred['confidence'], gt_valid,
                    )
                    iter_loss = iter_loss + self.conf_nll_weight * cnll_loss
                    c_metrics.update(cnll_metrics)

                if self.conf_valid_weight > 0:
                    cv_loss, cv_metrics = confidence_validity_loss(
                        pred['confidence'], gt_valid,
                    )
                    iter_loss = iter_loss + self.conf_valid_weight * cv_loss
                    c_metrics.update(cv_metrics)

                if self.conf_epe_bce_weight > 0:
                    cepe_loss, cepe_metrics = confidence_epe_bce_loss(
                        pred['flow'],
                        gt_flow,
                        pred['confidence'],
                        gt_valid,
                        good_px=self.conf_epe_good_px,
                        bad_px=self.conf_epe_bad_px,
                    )
                    iter_loss = iter_loss + self.conf_epe_bce_weight * cepe_loss
                    c_metrics.update(cepe_metrics)

                if (
                    self.corr_feat_weight > 0
                    and (self._map_decoder_active or self._map_fsm_active)
                    and fine_bundle.get('fine_features') is not None
                ):
                    corr_feat_loss, corr_feat_metrics = flow_correspondence_feature_loss(
                        fine_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                    )
                    iter_loss = iter_loss + self.corr_feat_weight * corr_feat_loss
                    c_metrics.update(corr_feat_metrics)

                if self.corr_ce_weight > 0 and fine_bundle.get('fine_features') is not None:
                    corr_ce_loss, corr_ce_metrics = local_correlation_ce_loss(
                        self.model,
                        fine_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                        radius=int(getattr(self.model, 'local_radius', 4)),
                        temperature=self.corr_ce_temperature,
                    )
                    iter_loss = iter_loss + self.corr_ce_weight * corr_ce_loss
                    c_metrics.update(corr_ce_metrics)

                if self.corr_subpixel_ce_weight > 0 and fine_bundle.get('fine_features') is not None:
                    corr_subpx_loss, corr_subpx_metrics = local_correlation_subpixel_ce_loss(
                        self.model,
                        fine_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                        radius=int(getattr(self.model, 'local_radius', 4)),
                        temperature=self.corr_subpixel_ce_temperature,
                    )
                    iter_loss = iter_loss + self.corr_subpixel_ce_weight * corr_subpx_loss
                    c_metrics.update(corr_subpx_metrics)

                if self.corr_flow_weight > 0 and fine_bundle.get('fine_features') is not None:
                    corr_flow_loss, corr_flow_metrics = local_correlation_soft_flow_loss(
                        self.model,
                        fine_bundle['fine_features'],
                        query_fine,
                        gt_flow,
                        gt_valid,
                        radius=int(getattr(self.model, 'local_radius', 4)),
                        temperature=self.corr_flow_temperature,
                    )
                    iter_loss = iter_loss + self.corr_flow_weight * corr_flow_loss
                    c_metrics.update(corr_flow_metrics)

                # Pose loss (Phase 2)
                p_metrics = {}
                if use_pose_loss:
                    p_loss, p_metrics = pose_loss(
                        pred['delta_xi'], pose_mid, pose_gt,
                        rot_weight=self.rot_weight,
                        trans_weight=self.trans_weight,
                        rot_loss_type=self.rot_loss_type,
                        loss_mode=self.pose_loss_mode,
                    )
                    iter_loss = iter_loss + self.pose_weight * p_loss

                    if self.full_pose_weight > 0 and 'delta_xi_full' in pred:
                        full_p_loss, full_p_metrics = pose_loss(
                            pred['delta_xi_full'], pose_mid, pose_gt,
                            rot_weight=self.rot_weight,
                            trans_weight=self.trans_weight,
                            rot_loss_type=self.rot_loss_type,
                            loss_mode=self.pose_loss_mode,
                        )
                        iter_loss = iter_loss + self.full_pose_weight * full_p_loss
                        p_metrics.update({
                            f'full_{k}': v for k, v in full_p_metrics.items()
                        })

                    # Direct translation supervision for regression head
                    if self.use_direct_trans_loss:
                        with torch.cuda.amp.autocast(enabled=False):
                            T_rel_gt = torch.bmm(pose_gt.float(), torch.inverse(pose_mid.float()))
                            gt_xi = se3_log(T_rel_gt)  # (B, 6) correct Lie algebra
                            gt_trans_lie = gt_xi[:, :3]
                            pred_trans = pred['delta_xi'][:, :3].float()
                            direct_trans_loss = F.l1_loss(pred_trans, gt_trans_lie)
                        iter_loss = iter_loss + self.trans_weight * direct_trans_loss
                        p_metrics['direct_trans_loss'] = direct_trans_loss.item()

            # NaN check
            if torch.isnan(iter_loss) or torch.isinf(iter_loss):
                return {'nan_step': True}

            # Weighted backward
            gamma_outer = 0.8
            iter_w = gamma_outer ** (N - 1 - outer_i)
            scaled_loss = iter_w * iter_loss / N
            self.scaler.scale(scaled_loss).backward()

            total_loss_val += iter_loss.item()

            # Record last iteration metrics
            if outer_i == N - 1:
                all_metrics = f_metrics.copy()
                all_metrics.update(c_metrics)
                all_metrics.update(coarse_metrics)
                all_metrics.update(p_metrics)
                all_metrics['ran_coarse_stage'] = float(iter_state['ran_coarse_stage'])

            # Update pose for next outer iteration
            if outer_i < N - 1:
                pose_cur = iter_state['pose_next'].detach()

        all_metrics['total_loss'] = total_loss_val / N
        all_metrics['outer_iters_used'] = float(N)

        # Auxiliary feature matching loss for decoder fine-tuning
        if (self._map_decoder_active or self._map_fsm_active) and self.feat_match_weight > 0:
            ref_gt = self._render_batch_differentiable(pose_gt)['fine_features']
            # Cosine similarity between rendered features at GT pose and query features
            ref_norm = F.normalize(ref_gt.float(), dim=1)
            q_norm = F.normalize(query_fine.float(), dim=1)
            cos_sim = (ref_norm * q_norm).sum(dim=1).mean()
            feat_match_loss = 1.0 - cos_sim
            self.scaler.scale(self.feat_match_weight * feat_match_loss).backward()
            all_metrics['feat_match_loss'] = feat_match_loss.item()
            all_metrics['feat_cos_sim'] = cos_sim.item()
            all_metrics['total_loss'] += self.feat_match_weight * feat_match_loss.item()

        if (
            (self._map_decoder_active or self._map_fsm_active)
            and self.pose_rank_weight > 0
            and self.epoch >= self.pose_rank_start_epoch
        ):
            rank_loss, rank_metrics = self._pose_perturb_rank_loss(
                query_fine,
                pose_gt,
            )
            self.scaler.scale(self.pose_rank_weight * rank_loss).backward()
            for k, v in rank_metrics.items():
                all_metrics[k] = v
            all_metrics['total_loss'] += self.pose_rank_weight * rank_loss.item()

        if (
            (self._map_decoder_active or self._map_fsm_active)
            and self.loc_rank_weight > 0
            and self.epoch >= self.loc_rank_start_epoch
        ):
            loc_rank_loss, loc_rank_metrics = self._loc_cm_perturb_rank_loss(
                query_fine,
                pose_gt,
            )
            self.scaler.scale(self.loc_rank_weight * loc_rank_loss).backward()
            for k, v in loc_rank_metrics.items():
                all_metrics[k] = v
            all_metrics['total_loss'] += self.loc_rank_weight * loc_rank_loss.item()

        if (
            (self._map_decoder_active or self._map_fsm_active)
            and self.feature_metric_weight > 0
            and self.epoch >= self.feature_metric_start_epoch
        ):
            fm_loss, fm_metrics = self._feature_metric_training_loss(
                query_fine,
                pose_start,
                pose_gt,
                render_intr,
            )
            self.scaler.scale(self.feature_metric_weight * fm_loss).backward()
            for k, v in fm_metrics.items():
                all_metrics[k] = v
            all_metrics['total_loss'] += self.feature_metric_weight * fm_loss.item()

        return all_metrics

    # ── Training Epoch ────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self._sync_map_finetune_state(epoch)
        self.model.train()
        self._set_map_train_mode(True)
        use_pose_loss = epoch >= self.phase1_epochs

        epoch_metrics = {}
        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch}/{self.total_epochs}',
                    leave=False)

        for batch_idx, batch in enumerate(pbar):
            if self.max_train_batches > 0 and batch_idx >= self.max_train_batches:
                break
            self.optimizer.zero_grad()
            metrics = self._train_step(batch, use_pose_loss)

            # NaN guard
            if metrics.get('nan_step'):
                self.global_step += 1
                continue

            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                trainable_params = [
                    p
                    for group in self.optimizer.param_groups
                    for p in group['params']
                    if p.requires_grad
                ]
                valid_grads = True
                for p in trainable_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        valid_grads = False
                        break
                if not valid_grads:
                    self.optimizer.zero_grad()
                    self.scaler.update()
                    self.global_step += 1
                    continue
                nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != 'nan_step':
                    epoch_metrics.setdefault(k, []).append(v)

            # Progress bar
            pbar_dict = {'loss': f"{metrics.get('total_loss', 0):.4f}"}
            if 'rot_err_deg' in metrics:
                pbar_dict['rot'] = f"{metrics['rot_err_deg']:.2f}°"
                pbar_dict['trans'] = f"{metrics['trans_err_mm']:.1f}mm"
            pbar.set_postfix(pbar_dict)

            # TensorBoard
            if self.global_step % 50 == 0:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k != 'nan_step':
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'],
                    self.global_step)

        # Epoch averages
        avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        if self.feature_only_train:
            phase = "Feature-only"
        else:
            phase = "Phase2(flow+pose)" if use_pose_loss else "Phase1(flow only)"
        msg = (f"[Train E{epoch}] {phase}  "
               f"loss={avg.get('total_loss', 0):.4f}  "
               f"flow={avg.get('flow_loss', 0):.4f}  "
               f"epe={avg.get('flow_epe', 0):.2f}")
        if 'rot_err_deg' in avg:
            msg += f"  rot={avg['rot_err_deg']:.2f}°  trans={avg['trans_err_mm']:.1f}mm"
        if 'feat_cos_sim' in avg:
            msg += f"  cos_sim={avg['feat_cos_sim']:.3f}"
        if 'corr_feat_cos' in avg:
            msg += f"  corr_cos={avg['corr_feat_cos']:.3f}"
        if 'corr_ce_acc' in avg:
            msg += f"  corr_acc={avg['corr_ce_acc']:.3f}"
        if 'corr_subpx_flow_epe' in avg:
            msg += f"  corr_subpx_epe={avg['corr_subpx_flow_epe']:.2f}"
        if 'corr_flow_epe' in avg:
            msg += f"  corr_flow_epe={avg['corr_flow_epe']:.2f}"
        if 'loc_rank_acc' in avg:
            msg += (
                f"  loc_rank={avg['loc_rank_acc']:.3f}/"
                f"gap={avg.get('loc_rank_gap', 0.0):.3f}"
            )
        if 'fm_trans_err_mm' in avg:
            msg += (
                f"  fm={avg.get('fm_rot_err_deg', 0.0):.2f}°/"
                f"{avg['fm_trans_err_mm']:.1f}mm"
            )
        outer_used = int(round(float(avg.get('outer_iters_used', self.outer_iters_train))))
        if outer_used > 1:
            msg += f"  ({outer_used} outer iters)"
        self.logger.info(msg)

        return avg

    # ── Validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation with multi-iteration refinement."""
        np_state = np.random.get_state()
        np.random.seed(self.val_seed)
        torch.cuda.empty_cache()
        self.model.eval()
        self._set_map_train_mode(False)
        all_rot_errs, all_trans_errs = [], []
        all_init_rot_errs, all_init_trans_errs = [], []
        all_one_rot_errs, all_one_trans_errs = [], []
        all_flow_epe = []
        all_corr_flow_epe = []
        all_fm_rot_errs, all_fm_trans_errs = [], []
        sample_records = []
        N = self.outer_iters_val
        eval_gru_iters = int(getattr(self.model, 'gru_iters', 0))
        eval_seed = int(getattr(self, '_eval_current_seed', -1))

        flow_hw = (self.render_h, self.render_w)
        render_intr = self.model._scale_intrinsics(self.render_h, self.render_w)

        def compute_pose_error(pose_est: torch.Tensor, pose_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.cuda.amp.autocast(enabled=False):
                R_est = pose_est.float()[:, :3, :3]
                R_ref = pose_ref.float()[:, :3, :3]
                R_rel = torch.bmm(R_est.transpose(1, 2), R_ref)
                trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                cos_angle = torch.clamp(
                    (trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                rot_err = torch.acos(cos_angle) * 180.0 / math.pi
                c_est = camera_centers_from_w2c(pose_est.float())
                c_ref = camera_centers_from_w2c(pose_ref.float())
                trans_err = torch.norm(c_est - c_ref, dim=1) * 1000
            return rot_err, trans_err

        def render_batch_fn(poses_w2c: torch.Tensor, render_coarse: bool | None) -> Dict[str, Optional[torch.Tensor]]:
            return self._render_bundle_batch(
                poses_w2c,
                differentiable=False,
                render_coarse=render_coarse,
            )

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validating', leave=False)):
            if self.max_val_batches > 0 and batch_idx >= self.max_val_batches:
                break
            query_fine = batch['query_fine'].to(self.device)
            query_coarse = batch.get('query_coarse')
            if query_coarse is not None and self.use_coarse:
                query_coarse = query_coarse.to(self.device)
            else:
                query_coarse = None
            pose_gt = batch['pose_gt'].to(self.device)
            pose_cur = batch['pose_init'].to(self.device)
            pose_init_for_eval = pose_cur
            image_ids = batch.get('image_id', [])
            image_names = batch.get('image_name', [])
            final_state = None
            pose_after_one = None

            init_rot, init_trans = compute_pose_error(pose_cur, pose_gt)
            all_init_rot_errs.extend(init_rot.cpu().tolist())
            all_init_trans_errs.extend(init_trans.cpu().tolist())

            if self.feature_metric_eval:
                fm_bundle = self._render_bundle_batch(
                    pose_init_for_eval,
                    differentiable=False,
                    render_coarse=False,
                )
                fm_mask = self._depth_alpha_mask(fm_bundle)
                fm_query, fm_rendered = self._localization_feature_pair(
                    query_fine,
                    fm_bundle['fine_features'],
                )
                _fm_delta, fm_pose, _fm_residual = feature_metric_pose_update(
                    fm_query,
                    fm_rendered,
                    fm_bundle['depth'],
                    pose_init_for_eval,
                    render_intr,
                    damping=self.feature_metric_damping,
                    valid_mask=fm_mask,
                    normalize_features=self.feature_metric_normalize and not self.loc_use_projection,
                    update_scale=self.feature_metric_update_scale,
                )
                fm_rot, fm_trans = compute_pose_error(fm_pose, pose_gt)
                all_fm_rot_errs.extend(fm_rot.cpu().tolist())
                all_fm_trans_errs.extend(fm_trans.cpu().tolist())

            # Outer iteration refinement
            for outer_i in range(N):
                final_state = run_model_refine_iteration(
                    self.model,
                    render_batch_fn,
                    query_fine,
                    query_coarse,
                    pose_cur,
                    render_intr,
                    outer_iter=outer_i,
                    autocast_enabled=self.use_amp,
                )
                if outer_i == 0:
                    pose_after_one = final_state['pose_next']
                if outer_i < N - 1:
                    pose_cur = final_state['pose_next']

            if final_state is None:
                continue

            pred = final_state['fine_pred']
            depth = final_state['fine_bundle']['depth']
            pose_mid = final_state['pose_mid']
            pose_pred = final_state['pose_next']

            # Flow EPE on final iteration
            gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
                pose_mid, pose_gt, depth, flow_hw, render_intr,
            )
            epe_map = torch.norm(
                pred['flow'].float() - gt_flow.float(),
                dim=1, keepdim=True,
            )
            n_valid = gt_valid.sum().clamp(min=1.0)
            epe = (epe_map * gt_valid).sum() / n_valid
            all_flow_epe.append(epe.item())

            if self.corr_flow_weight > 0 and final_state['fine_bundle'].get('fine_features') is not None:
                _corr_loss, corr_metrics = local_correlation_soft_flow_loss(
                    self.model,
                    final_state['fine_bundle']['fine_features'],
                    query_fine,
                    gt_flow,
                    gt_valid,
                    radius=int(getattr(self.model, 'local_radius', 4)),
                    temperature=self.corr_flow_temperature,
                )
                all_corr_flow_epe.append(float(corr_metrics.get('corr_flow_epe', 0.0)))

            # Pose evaluation
            if 'delta_xi' in pred:
                if pose_after_one is not None:
                    one_rot, one_trans = compute_pose_error(pose_after_one, pose_gt)
                    all_one_rot_errs.extend(one_rot.cpu().tolist())
                    all_one_trans_errs.extend(one_trans.cpu().tolist())

                rot_err, trans_err = compute_pose_error(pose_pred, pose_gt)

                all_rot_errs.extend(rot_err.cpu().tolist())
                all_trans_errs.extend(trans_err.cpu().tolist())

                if bool(getattr(self, '_eval_collect_samples', False)):
                    if pose_after_one is not None:
                        one_rot, one_trans = compute_pose_error(pose_after_one, pose_gt)
                    else:
                        one_rot, one_trans = init_rot, init_trans
                    epe_per_sample = (
                        (epe_map * gt_valid).sum(dim=(1, 2, 3))
                        / gt_valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    )
                    B = int(rot_err.shape[0])
                    for bi in range(B):
                        image_id = image_ids[bi] if isinstance(image_ids, list) else image_ids[bi].item()
                        image_name = ''
                        if isinstance(image_names, list) and bi < len(image_names):
                            image_name = str(image_names[bi])
                        sample_records.append(
                            build_eval_sample_record(
                                image_id=int(image_id),
                                image_name=image_name,
                                outer_iters=N,
                                gru_iters=eval_gru_iters,
                                seed=eval_seed,
                                init_rot_deg=float(init_rot[bi].item()),
                                init_trans_mm=float(init_trans[bi].item()),
                                one_rot_deg=float(one_rot[bi].item()),
                                one_trans_mm=float(one_trans[bi].item()),
                                final_rot_deg=float(rot_err[bi].item()),
                                final_trans_mm=float(trans_err[bi].item()),
                                flow_epe_px=float(epe_per_sample[bi].item()),
                            )
                        )

        val_metrics = {}
        if all_rot_errs:
            rot = np.array(all_rot_errs)
            trans = np.array(all_trans_errs)

            joint_01_53 = float(np.mean((rot < 0.1) & (trans < 5.3)) * 100)
            joint_1_50 = float(np.mean((rot < 1.0) & (trans < 50.0)) * 100)
            joint_5_100 = float(np.mean((rot < 5.0) & (trans < 100.0)) * 100)

            val_metrics = {
                'val_rot_mean': float(np.nanmean(rot)),
                'val_rot_median': float(np.nanmedian(rot)),
                'val_trans_mean': float(np.nanmean(trans)),
                'val_trans_median': float(np.nanmedian(trans)),
                'val_pct_1deg': float(np.mean(rot < 1.0) * 100),
                'val_pct_5deg': float(np.mean(rot < 5.0) * 100),
                'val_joint_01deg_53mm': joint_01_53,
                'val_joint_1deg_50mm': joint_1_50,
                'val_joint_5deg_100mm': joint_5_100,
            }
            if all_init_rot_errs:
                init_rot = np.array(all_init_rot_errs)
                init_trans = np.array(all_init_trans_errs)
                val_metrics.update({
                    'val_init_rot_mean': float(np.nanmean(init_rot)),
                    'val_init_rot_median': float(np.nanmedian(init_rot)),
                    'val_init_trans_mean': float(np.nanmean(init_trans)),
                    'val_init_trans_median': float(np.nanmedian(init_trans)),
                })
            if all_one_rot_errs:
                one_rot = np.array(all_one_rot_errs)
                one_trans = np.array(all_one_trans_errs)
                val_metrics.update({
                    'val_one_rot_mean': float(np.nanmean(one_rot)),
                    'val_one_rot_median': float(np.nanmedian(one_rot)),
                    'val_one_trans_mean': float(np.nanmean(one_trans)),
                    'val_one_trans_median': float(np.nanmedian(one_trans)),
                })
            if all_flow_epe:
                val_metrics['val_flow_epe'] = float(np.mean(all_flow_epe))
            if all_corr_flow_epe:
                val_metrics['val_corr_flow_epe'] = float(np.mean(all_corr_flow_epe))
            if all_fm_rot_errs:
                fm_rot = np.array(all_fm_rot_errs)
                fm_trans = np.array(all_fm_trans_errs)
                val_metrics.update({
                    'val_fm_rot_mean': float(np.nanmean(fm_rot)),
                    'val_fm_rot_median': float(np.nanmedian(fm_rot)),
                    'val_fm_trans_mean': float(np.nanmean(fm_trans)),
                    'val_fm_trans_median': float(np.nanmedian(fm_trans)),
                })

            iters_str = f'  ({N} iters)' if N > 1 else ''
            stage_str = ''
            if 'val_init_trans_median' in val_metrics and 'val_one_trans_median' in val_metrics:
                stage_str = (
                    f'  init_med={val_metrics["val_init_rot_median"]:.2f}°/'
                    f'{val_metrics["val_init_trans_median"]:.1f}mm'
                    f'  one_med={val_metrics["val_one_rot_median"]:.2f}°/'
                    f'{val_metrics["val_one_trans_median"]:.1f}mm'
                )
            flow_str = ''
            if 'val_flow_epe' in val_metrics:
                flow_str = f'  flow_epe={val_metrics["val_flow_epe"]:.2f}px'
            if 'val_corr_flow_epe' in val_metrics:
                flow_str += f'  corr_epe={val_metrics["val_corr_flow_epe"]:.2f}px'
            fm_str = ''
            if 'val_fm_trans_median' in val_metrics:
                fm_str = (
                    f'  fm_med={val_metrics["val_fm_rot_median"]:.2f}°/'
                    f'{val_metrics["val_fm_trans_median"]:.1f}mm'
                )
            self.logger.info(
                f'[Val E{epoch}]  rot={val_metrics["val_rot_mean"]:.2f}° '
                f'(med {val_metrics["val_rot_median"]:.2f}°)  '
                f'trans={val_metrics["val_trans_mean"]:.1f}mm '
                f'(med {val_metrics["val_trans_median"]:.1f}mm)  '
                f'<1°={val_metrics["val_pct_1deg"]:.1f}%  '
                f'joint@1°/50mm={joint_1_50:.1f}%'
                f'{stage_str}'
                f'{flow_str}'
                f'{fm_str}'
                f'{iters_str}'
            )

            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)
        if bool(getattr(self, '_eval_collect_samples', False)):
            self._eval_sample_records = sample_records

        self.model.train()
        self._set_map_train_mode(True)
        np.random.set_state(np_state)
        return val_metrics

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_trans': self.best_val_trans,
            'best_metric_name': getattr(self, 'best_metric_name', 'trans_median'),
            'best_metric_value': getattr(self, 'best_metric_value', self.best_val_trans),
            'best_metric_label': getattr(self, 'best_metric_label', 'val_trans_median'),
            'config': self.config,
        }
        dcff_renderer = getattr(self, 'dcff_renderer', None)
        if dcff_renderer is not None and getattr(dcff_renderer, 'fine_decoder', None) is not None:
            ckpt['fine_decoder_state'] = dcff_renderer.fine_decoder.state_dict()
            if getattr(self, 'feat_sharp_fine', None) is not None:
                ckpt['feat_sharp_state'] = self.feat_sharp_fine.state_dict()
            if getattr(dcff_renderer, 'coarse_carrier_fusion', None) is not None:
                ckpt['coarse_fusion_state'] = dcff_renderer.coarse_carrier_fusion.state_dict()
        if getattr(self, 'feat_select', None) is not None:
            ckpt['fsm_state'] = self.feat_select.state_dict()
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
        if self.save_epoch_checkpoints and self.save_every > 0 and epoch % self.save_every == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:03d}.pth')

    def _cleanup_epoch_checkpoints(self):
        for ckpt_path in self.ckpt_dir.glob('epoch_*.pth'):
            if ckpt_path.is_file():
                ckpt_path.unlink()

    def _load_checkpoint(self, path: str):
        self.logger.info(f'[Resume] Loading from {path}')
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.epoch = ckpt['epoch'] + 1
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_trans = ckpt.get('best_val_trans', float('inf'))
        self.best_metric_name = ckpt.get('best_metric_name', self.best_metric_name)
        self.best_metric_value = ckpt.get('best_metric_value', self.best_metric_value)
        self.best_metric_label = ckpt.get('best_metric_label', self.best_metric_label)

        # Handle epoch extension
        old_epochs = ckpt.get('config', {}).get('training', {}).get(
            'epochs', self.total_epochs)
        if self.total_epochs > old_epochs and self.epoch >= old_epochs:
            remaining = self.total_epochs - self.epoch
            tc = self.config.get('training', {})
            ext_lr = tc.get('lr', 3e-4) * 0.1
            for pg in self.optimizer.param_groups:
                pg['lr'] = ext_lr
                pg['initial_lr'] = ext_lr
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(remaining, 1),
                eta_min=tc.get('min_lr', 1e-6),
            )
            self.logger.info(
                f'  Extended training: fresh cosine LR={ext_lr} '
                f'over {remaining} epochs')
        else:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # Restore the exact DCFF feature runtime used by the checkpoint.  This
        # matters even when the decoder/FSM are frozen in the resumed config.
        if 'fine_decoder_state' in ckpt and hasattr(self, 'dcff_renderer'):
            self.dcff_renderer.fine_decoder.load_state_dict(
                ckpt['fine_decoder_state'])
            self.logger.info('  Restored checkpoint decoder weights')
        if (
            'coarse_fusion_state' in ckpt
            and hasattr(self, 'dcff_renderer')
            and self.dcff_renderer.coarse_carrier_fusion is not None
        ):
            self.dcff_renderer.coarse_carrier_fusion.load_state_dict(
                ckpt['coarse_fusion_state'])
            self.logger.info('  Restored checkpoint coarse fusion weights')
        if 'feat_sharp_state' in ckpt and hasattr(self, 'feat_sharp_fine'):
            self.feat_sharp_fine.load_state_dict(
                ckpt['feat_sharp_state'])
            self.logger.info('  Restored checkpoint feat_sharp weights')
        if self.feat_select is not None and 'fsm_state' in ckpt:
            self.feat_select.load_state_dict(ckpt['fsm_state'])
            self.logger.info('  Restored checkpoint FSM weights')

        self.logger.info(f'  Resumed at epoch {self.epoch}, step {self.global_step}')

    def _warmstart(self, path: str):
        """Load model weights only, keep fresh optimizer/scheduler."""
        self.logger.info(f'[Warmstart] Loading model weights from {path}')
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt['model_state_dict']

        model_state = self.model.state_dict()
        filtered_state = {}
        skip_prefixes = tuple(self.warmstart_skip_prefixes)
        skipped_by_prefix = []
        skipped = []
        for k, v in state_dict.items():
            if skip_prefixes and any(k.startswith(prefix) for prefix in skip_prefixes):
                skipped_by_prefix.append(k)
            elif k in model_state and v.shape != model_state[k].shape:
                skipped.append(
                    f'{k}: ckpt {list(v.shape)} vs model {list(model_state[k].shape)}')
            else:
                filtered_state[k] = v

        if skipped_by_prefix:
            self.logger.info(
                '  Skipped %d warmstart keys by prefix: %s',
                len(skipped_by_prefix),
                ', '.join(sorted(skip_prefixes)),
            )

        if skipped:
            self.logger.warning(f'  Skipped {len(skipped)} size-mismatched keys:')
            for s in skipped:
                self.logger.warning(f'    {s}')

        missing, unexpected = self.model.load_state_dict(
            filtered_state, strict=False)
        if missing:
            self.logger.warning(f'  Missing keys: {missing}')
        if unexpected:
            self.logger.warning(f'  Unexpected keys: {unexpected}')

        # Restore fine-tuned decoder weights if present in warmstart checkpoint
        if 'fine_decoder_state' in ckpt and hasattr(self, 'dcff_renderer'):
            self.dcff_renderer.fine_decoder.load_state_dict(
                ckpt['fine_decoder_state'])
            self.logger.info('  Restored fine-tuned decoder weights from warmstart')
        if ('coarse_fusion_state' in ckpt and hasattr(self, 'dcff_renderer')
                and self.dcff_renderer.coarse_carrier_fusion is not None):
            self.dcff_renderer.coarse_carrier_fusion.load_state_dict(
                ckpt['coarse_fusion_state'])
            self.logger.info('  Restored coarse fusion weights from warmstart')
        if 'feat_sharp_state' in ckpt and hasattr(self, 'feat_sharp_fine'):
            self.feat_sharp_fine.load_state_dict(ckpt['feat_sharp_state'])
            self.logger.info('  Restored feat_sharp weights from warmstart')
        if 'fsm_state' in ckpt and getattr(self, 'feat_select', None) is not None:
            self.feat_select.load_state_dict(ckpt['fsm_state'])
            self.logger.info('  Restored FSM weights from warmstart')

        src_epoch = ckpt.get('epoch', '?')
        self.logger.info(f'  Loaded weights from epoch {src_epoch}, fresh optimizer')

    # ── Visualization ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _visualize(self, epoch: int):
        """Generate feature + flow + confidence visualizations."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        vis_dir = self.vis_dir / f'epoch_{epoch:04d}'
        vis_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self._set_map_train_mode(False)
        torch.cuda.empty_cache()

        batch = next(iter(self.val_loader))
        n_vis = min(self.num_vis_samples, batch['query_fine'].shape[0])
        query_fine = batch['query_fine'][:n_vis].to(self.device)
        query_coarse = batch.get('query_coarse')
        if query_coarse is not None and self.use_coarse:
            query_coarse = query_coarse[:n_vis].to(self.device)
        else:
            query_coarse = None
        pose_gt = batch['pose_gt'][:n_vis].to(self.device)
        pose_init = batch['pose_init'][:n_vis].to(self.device)

        render_intr = self.model._scale_intrinsics(self.render_h, self.render_w)

        def render_batch_fn(poses_w2c: torch.Tensor, render_coarse: bool | None) -> Dict[str, Optional[torch.Tensor]]:
            return self._render_bundle_batch(
                poses_w2c,
                differentiable=False,
                render_coarse=render_coarse,
            )

        iter_state = run_model_refine_iteration(
            self.model,
            render_batch_fn,
            query_fine,
            query_coarse,
            pose_init,
            render_intr,
            outer_iter=0,
            autocast_enabled=self.use_amp,
        )
        pred = iter_state['fine_pred']
        ref_fine = iter_state['fine_bundle']['fine_features']
        depth = iter_state['fine_bundle']['depth']
        ref_coarse = None
        fsm_spatial = None
        if iter_state['coarse_bundle'] is not None:
            ref_coarse = iter_state['coarse_bundle'].get('coarse_features')
            fsm_spatial = iter_state['coarse_bundle'].get('fsm_spatial_conf')
        if fsm_spatial is None:
            fsm_spatial = iter_state['fine_bundle'].get('fsm_spatial_conf')

        # (1) PCA Feature Visualization
        fig, axes = plt.subplots(n_vis, 2, figsize=(8, 4 * n_vis))
        if n_vis == 1:
            axes = axes[np.newaxis, :]
        for i in range(n_vis):
            q_pca = features_to_pca_rgb(query_fine[i].cpu())
            r_pca = features_to_pca_rgb(ref_fine[i].cpu())

            axes[i, 0].imshow(q_pca.permute(1, 2, 0).numpy())
            axes[i, 0].set_title('Query Fine' if i == 0 else '', fontsize=8)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(r_pca.permute(1, 2, 0).numpy())
            axes[i, 1].set_title('Rendered Fine' if i == 0 else '', fontsize=8)
            axes[i, 1].axis('off')

        fig.suptitle(f'Feature PCA (Epoch {epoch})', fontsize=11)
        fig.tight_layout()
        fig.savefig(str(vis_dir / 'features_pca.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        if ref_coarse is not None:
            fig, axes = plt.subplots(n_vis, 2, figsize=(8, 4 * n_vis))
            if n_vis == 1:
                axes = axes[np.newaxis, :]
            for i in range(n_vis):
                q_src = query_coarse[i].cpu() if query_coarse is not None else query_fine[i].cpu()
                q_pca = features_to_pca_rgb(q_src)
                r_pca = features_to_pca_rgb(ref_coarse[i].cpu())

                axes[i, 0].imshow(q_pca.permute(1, 2, 0).numpy())
                axes[i, 0].set_title('Query Coarse' if i == 0 else '', fontsize=8)
                axes[i, 0].axis('off')
                axes[i, 1].imshow(r_pca.permute(1, 2, 0).numpy())
                axes[i, 1].set_title('Rendered Coarse' if i == 0 else '', fontsize=8)
                axes[i, 1].axis('off')

            fig.suptitle(f'Coarse Feature PCA (Epoch {epoch})', fontsize=11)
            fig.tight_layout()
            fig.savefig(str(vis_dir / 'coarse_features_pca.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        # (2) Flow Visualization
        flow = pred['flow'][0].cpu().float()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        vmax = max(flow.abs().max().item(), 1.0)
        im0 = axes[0].imshow(flow[0].numpy(), cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        axes[0].set_title(f'Flow u E{epoch}', fontsize=9)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(flow[1].numpy(), cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Flow v E{epoch}', fontsize=9)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(vis_dir / 'flow.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # (3) Confidence Map
        conf = pred['confidence'][0].cpu().float()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ch, label in enumerate(['u', 'v']):
            im = axes[ch].imshow(conf[ch].numpy(), cmap='viridis', vmin=0, vmax=1)
            axes[ch].set_title(
                f'Conf {label} E{epoch} mean={conf[ch].mean():.3f}', fontsize=9)
            axes[ch].axis('off')
            plt.colorbar(im, ax=axes[ch], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(vis_dir / 'confidence.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # (4) Depth Visualization
        d = depth[0].cpu().float().numpy()
        d_valid = d[d > 0.05]
        if len(d_valid) > 0:
            d_lo, d_hi = np.percentile(d_valid, [2, 98])
            d_norm = np.clip((d - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
        else:
            d_norm = np.zeros_like(d)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(d_norm, cmap='turbo')
        ax.set_title(f'Depth E{epoch}', fontsize=9)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(str(vis_dir / 'depth.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        if fsm_spatial is not None:
            conf_map = fsm_spatial[0].cpu().float().squeeze(0).numpy()
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            im = ax.imshow(conf_map, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'FSM Spatial Confidence E{epoch}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(str(vis_dir / 'fsm_spatial_conf.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        # TensorBoard images
        try:
            import torchvision.transforms.functional as TF
            from PIL import Image
            for fname in ['features_pca.png', 'coarse_features_pca.png', 'fsm_spatial_conf.png', 'flow.png', 'confidence.png',
                          'depth.png']:
                fpath = vis_dir / fname
                if fpath.exists():
                    img = Image.open(str(fpath))
                    self.writer.add_image(
                        f'vis/{fname.replace(".png", "")}',
                        TF.to_tensor(img), epoch,
                    )
        except Exception:
            pass

        self.logger.info(f'  [Vis E{epoch}] Saved to {vis_dir}')
        self.model.train()
        self._set_map_train_mode(True)
        torch.cuda.empty_cache()

    # ── Main Training Loop ────────────────────────────────────────────────

    def train(self):
        """Full training loop."""
        self.logger.info(f'\n{"="*60}')
        self.logger.info(f'  Concat-Based Localization Training')
        self.logger.info(f'  Epochs: {self.total_epochs} '
                         f'(Phase1: {self.phase1_epochs})')
        self.logger.info(f'  Outer iters: {self.outer_iters_train} train / '
                         f'{self.outer_iters_val} val')
        self.logger.info(f'  Output: {self.output_dir}')
        self.logger.info(f'{"="*60}\n')

        epoch_times = []
        train_start = time.time()

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch
            t0 = time.time()

            # Noise curriculum
            self._update_noise_for_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = {}
            if epoch % self.val_every == 0 or epoch == self.total_epochs - 1:
                val_metrics = self.validate(epoch)
                self.latest_val_metrics = dict(val_metrics)

            # Scheduler step
            self.scheduler.step()

            # Checkpoint — default to median translation, but allow rotation/joint
            # metrics for later localization-focused stages.
            is_best = False
            if val_metrics:
                med_trans = val_metrics.get('val_trans_median', float('inf'))
                if med_trans < self.best_val_trans:
                    self.best_val_trans = med_trans
                current_best_value, higher_is_better, metric_label = resolve_best_metric_value(
                    val_metrics,
                    self.best_metric_name,
                )
                improved = (
                    current_best_value > self.best_metric_value
                    if higher_is_better
                    else current_best_value < self.best_metric_value
                )
                if improved:
                    self.best_metric_value = current_best_value
                    self.best_metric_label = metric_label
                    self.epochs_since_best = 0
                    is_best = True
                    self.logger.info(
                        f'  ★ New best {metric_label}={current_best_value:.3f}: '
                        f'trans_med={med_trans:.1f}mm  '
                        f'rot_med={val_metrics.get("val_rot_median", 0):.2f}°  '
                        f'joint@1°/50mm='
                        f'{val_metrics.get("val_joint_1deg_50mm", 0):.1f}%'
                    )
                else:
                    self.epochs_since_best += 1

            self._save_checkpoint(epoch, is_best)

            # Early stopping
            if (self.early_stop_patience > 0 and
                    self.epochs_since_best >= self.early_stop_patience):
                self.logger.info(
                    f'  ⏹ Early stopping: {self.epochs_since_best} epochs '
                    f'without improvement (best={self.best_val_trans:.1f}mm)'
                )
                break

            # Periodic visualization
            if self.vis_every > 0 and (
                epoch % self.vis_every == 0 or epoch == self.total_epochs - 1
            ):
                self._visualize(epoch)

            # Timing
            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            avg_epoch = np.mean(epoch_times[-5:])
            remaining = (self.total_epochs - epoch - 1) * avg_epoch
            eta_min = remaining / 60
            total_elapsed = (time.time() - train_start) / 60
            self.logger.info(
                f'  Epoch {epoch}: {elapsed:.0f}s  '
                f'lr={self.optimizer.param_groups[0]["lr"]:.6f}  '
                f'ETA: {eta_min:.0f}min  '
                f'[{total_elapsed:.0f}min elapsed]\n'
            )

        total_time = (time.time() - train_start) / 60
        self.logger.info(
            f'\nTraining complete in {total_time:.0f}min! '
            f'Best val trans_med: {self.best_val_trans:.1f}mm'
        )

        final_metrics = {
            'best_val_trans_median': float(self.best_val_trans),
            'best_metric_name': str(self.best_metric_name),
            'best_metric_value': float(self.best_metric_value),
            'best_metric_label': str(self.best_metric_label),
            'epochs_completed': int(self.epoch + 1),
            'global_step': int(self.global_step),
            'total_time_min': float(total_time),
            'train_dataset_size': len(self.train_dataset),
            'val_dataset_size': len(self.val_dataset),
            'outer_iters_train': int(self.outer_iters_train),
            'outer_iters_val': int(self.outer_iters_val),
            'max_train_batches': int(self.max_train_batches),
            'max_val_batches': int(self.max_val_batches),
            'gru_iters': int(self.model.gru_iters),
            'use_coarse': bool(self.use_coarse),
            'use_two_stage_refine': bool(getattr(self.model, 'use_two_stage_refine', False)),
            'finetune_decoder': bool(self.finetune_decoder),
            'finetune_fsm': bool(getattr(self, 'finetune_fsm', False)),
            'feature_only_train': bool(self.feature_only_train),
        }
        if hasattr(self, 'latest_val_metrics') and self.latest_val_metrics:
            final_metrics['latest_val'] = dict(self.latest_val_metrics)

        summary_lines = [
            f'best val trans={self.best_val_trans:.1f}mm',
            f'best {self.best_metric_label}={self.best_metric_value:.3f}',
            f'epochs={self.epoch + 1} steps={self.global_step}',
            f'use_coarse={self.use_coarse} two_stage={bool(getattr(self.model, "use_two_stage_refine", False))} finetune_fsm={bool(getattr(self, "finetune_fsm", False))}',
        ]
        if hasattr(self, 'latest_val_metrics') and self.latest_val_metrics:
            summary_lines.append(
                'latest val rot={:.2f}deg trans={:.1f}mm joint@1/50={:.1f}%'.format(
                    float(self.latest_val_metrics.get('val_rot_median', 0.0)),
                    float(self.latest_val_metrics.get('val_trans_median', 0.0)),
                    float(self.latest_val_metrics.get('val_joint_1deg_50mm', 0.0)),
                )
            )

        notes = [
            f'config={self.output_dir / "config.yaml"}',
            f'gpu={self.device}',
        ]
        artifact_paths = [
            self.output_dir / 'config.yaml',
            self.ckpt_dir / 'best.pth',
            self.ckpt_dir / 'latest.pth',
            self.vis_dir,
            self.output_dir / f'{self.exp_name}_train.log',
        ]
        save_experiment_bundle(
            exp_name=self.exp_name,
            output_dir=self.output_dir,
            metrics=final_metrics,
            summary_lines=summary_lines,
            notes=notes,
            artifact_paths=artifact_paths,
            results_json_name='results.json',
            results_text_name='results.txt',
            report_markdown_name='report.md',
            report_text_name='report.txt',
        )
        self.writer.close()


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train Concat-Based Localization Network')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (full state)')
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Warmstart from checkpoint (model weights only)')
    parser.add_argument('--dcff_checkpoint', type=str, default=None,
                        help='Override DCFF checkpoint path')
    parser.add_argument('--localization_manifest', type=str, default=None,
                        help='Apply exported query/DCFF/init-cache manifest overrides')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only (no training)')
    parser.add_argument('--eval_outer_iters', type=int, nargs='+', default=None,
                        help='Override outer iters for eval sweep')
    parser.add_argument('--eval_gru_iters', type=int, nargs='+', default=None,
                        help='Override GRU iters for eval sweep')
    parser.add_argument('--eval_seeds', type=int, default=1,
                        help='Number of random seeds for eval averaging')
    parser.add_argument('--eval_sample_json', type=str, default=None,
                        help='Optional path for per-sample eval records JSON')
    args = parser.parse_args()

    config = load_mainline_config(args.config, localization_manifest=args.localization_manifest)

    trainer = ConcatLocTrainer(
        config=config,
        gpu=args.gpu,
        resume_path=args.resume,
        warmstart_path=args.warmstart,
        dcff_checkpoint=args.dcff_checkpoint,
    )

    if args.eval_only:
        # Eval sweep mode with optional multi-seed averaging
        outer_list = args.eval_outer_iters or [trainer.outer_iters_val]
        gru_list = args.eval_gru_iters or [trainer.model.gru_iters]
        n_seeds = args.eval_seeds
        orig_gru = trainer.model.gru_iters
        orig_outer = trainer.outer_iters_val
        sample_records = []
        trainer._eval_collect_samples = bool(args.eval_sample_json)
        print(f"\n{'outer':>6} {'gru':>4} | {'rot_med':>8} {'trans_med':>10} {'<1°':>6} {'j@1/50':>7}"
              + (f" (avg {n_seeds} seeds)" if n_seeds > 1 else ""))
        print("-" * 60)
        sweep_records = []
        for outer in outer_list:
            for gru in gru_list:
                trainer.outer_iters_val = outer
                trainer.model.gru_iters = gru
                all_rm, all_tm, all_p1, all_j1 = [], [], [], []
                for seed in range(n_seeds):
                    torch.manual_seed(42 + seed)
                    np.random.seed(42 + seed)
                    trainer._eval_current_seed = seed
                    trainer._eval_sample_records = []
                    metrics = trainer.validate(epoch=0)
                    if args.eval_sample_json:
                        sample_records.extend(getattr(trainer, '_eval_sample_records', []))
                    all_rm.append(metrics.get('val_rot_median', 0))
                    all_tm.append(metrics.get('val_trans_median', 0))
                    all_p1.append(metrics.get('val_pct_1deg', 0))
                    all_j1.append(metrics.get('val_joint_1deg_50mm', 0))
                rm = np.mean(all_rm)
                tm = np.mean(all_tm)
                p1 = np.mean(all_p1)
                j1 = np.mean(all_j1)
                avg_metrics = {
                    'val_rot_median': rm,
                    'val_trans_median': tm,
                    'val_pct_1deg': p1,
                    'val_joint_1deg_50mm': j1,
                }
                sweep_records.append(
                    build_eval_sweep_record(
                        outer_iters=outer,
                        gru_iters=gru,
                        metrics=avg_metrics,
                        seed_count=n_seeds,
                    )
                )
                if n_seeds > 1:
                    print(f"{outer:>6} {gru:>4} | {rm:>7.2f}° {tm:>9.1f}mm {p1:>5.1f}% {j1:>6.1f}%"
                          f"  (±{np.std(all_tm):.1f}mm)")
                else:
                    print(f"{outer:>6} {gru:>4} | {rm:>7.2f}° {tm:>9.1f}mm {p1:>5.1f}% {j1:>6.1f}%")
        sweep_path = trainer.output_dir / 'eval_sweep_results.json'
        with open(sweep_path, 'w', encoding='utf-8') as f:
            json.dump({'records': sweep_records}, f, indent=2)
        print(f"Eval sweep results saved to {sweep_path}")
        if args.eval_sample_json:
            sample_path = Path(args.eval_sample_json)
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump({'records': sample_records}, f, indent=2)
            print(f"Per-sample eval records saved to {sample_path}")
        trainer.model.gru_iters = orig_gru
        trainer.outer_iters_val = orig_outer
    else:
        trainer.train()


if __name__ == '__main__':
    main()
