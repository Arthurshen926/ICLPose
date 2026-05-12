"""Pose-conditioned neural energy and residual heads for CPR."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_refine.utils.lie_algebra import pose_inverse, se3_exp, se3_log


class PoseEnergyNet(nn.Module):
    """Score and update rendered pose candidates from query-map evidence.

    The network intentionally consumes candidate-level score maps plus vector
    priors, but unlike the previous selector it also predicts a 6D left-update
    residual for each candidate.
    """

    expects_score_map = True

    def __init__(
        self,
        vector_dim: int,
        score_map_channels: int = 3,
        map_channels: int = 16,
        grid_size: int = 4,
        hidden_dim: int = 128,
        context_layers: int = 1,
        context_heads: int = 1,
        context_feedforward_dim: int | None = None,
        dropout: float = 0.0,
        zero_init_heads: bool = False,
    ):
        super().__init__()
        self.vector_dim = int(vector_dim)
        self.score_map_channels = int(score_map_channels)
        self.map_channels = int(map_channels)
        self.grid_size = int(grid_size)
        self.hidden_dim = int(hidden_dim)
        self.context_layers = int(context_layers)
        self.context_heads = int(context_heads)
        if self.vector_dim < 0:
            raise ValueError("vector_dim must be non-negative")
        if self.score_map_channels <= 0:
            raise ValueError("score_map_channels must be positive")
        if self.map_channels <= 0:
            raise ValueError("map_channels must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.context_layers < 0:
            raise ValueError("context_layers must be non-negative")
        if self.context_heads <= 0:
            raise ValueError("context_heads must be positive")

        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.score_map_channels, self.map_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.map_channels, self.map_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        pooled_dim = self.map_channels * self.grid_size * self.grid_size
        input_dim = pooled_dim + self.vector_dim
        self.input_dim = input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
        )
        if self.context_layers > 0:
            if self.hidden_dim % self.context_heads != 0:
                raise ValueError(
                    f"context_heads={self.context_heads} must divide hidden_dim={self.hidden_dim}"
                )
            ff_dim = int(context_feedforward_dim or max(self.hidden_dim * 4, self.hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.context_heads,
                dim_feedforward=ff_dim,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=self.context_layers)
        else:
            self.context_encoder = None
        self.energy_head = nn.Linear(self.hidden_dim, 1)
        self.residual_head = nn.Linear(self.hidden_dim, 6)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)
        if bool(zero_init_heads):
            for head in (self.energy_head, self.residual_head, self.confidence_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(
        self,
        score_maps: torch.Tensor,
        vector_features: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if score_maps.ndim != 5:
            raise ValueError("score_maps must have shape (B,K,C,H,W)")
        if score_maps.shape[2] != self.score_map_channels:
            raise ValueError(
                f"expected score_map_channels={self.score_map_channels}, got {score_maps.shape[2]}"
            )
        bsz, num_candidates, channels, height, width = score_maps.shape
        flat_maps = score_maps.float().reshape(bsz * num_candidates, channels, height, width)
        encoded = self.map_encoder(flat_maps)
        pooled = F.adaptive_avg_pool2d(encoded, (self.grid_size, self.grid_size)).flatten(1)
        if self.vector_dim > 0:
            if vector_features is None:
                raise ValueError("vector_features are required when vector_dim > 0")
            if vector_features.shape != (bsz, num_candidates, self.vector_dim):
                raise ValueError(
                    f"vector_features must have shape {(bsz, num_candidates, self.vector_dim)}, "
                    f"got {tuple(vector_features.shape)}"
                )
            vector_flat = vector_features.float().reshape(bsz * num_candidates, self.vector_dim)
            tokens = torch.cat([pooled, vector_flat], dim=1)
        else:
            tokens = pooled
        tokens = self.input_proj(tokens).reshape(bsz, num_candidates, self.hidden_dim)
        if self.context_encoder is not None:
            tokens = self.context_encoder(tokens)
        energy_logits = self.energy_head(tokens).squeeze(-1).float()
        residual_delta = self.residual_head(tokens).float()
        confidence_logits = self.confidence_head(tokens).squeeze(-1).float()
        if valid_mask is not None:
            valid = valid_mask.to(device=energy_logits.device).bool()
            if valid.shape != (bsz, num_candidates):
                raise ValueError(f"valid_mask must have shape {(bsz, num_candidates)}")
            energy_logits = energy_logits.masked_fill(~valid, -1.0e6)
            confidence_logits = confidence_logits.masked_fill(~valid, -1.0e6)
            residual_delta = residual_delta.masked_fill(~valid[..., None], 0.0)
        return {
            "energy_logits": energy_logits,
            "residual_delta": residual_delta,
            "confidence_logits": confidence_logits,
            "tokens": tokens,
        }


def _camera_centers_from_w2c(pose_w2c: torch.Tensor) -> torch.Tensor:
    rot = pose_w2c[..., :3, :3]
    trans = pose_w2c[..., :3, 3]
    return -(rot.transpose(-1, -2) @ trans.unsqueeze(-1)).squeeze(-1)


def _rotation_error_rad(rot_pred: torch.Tensor, rot_gt: torch.Tensor) -> torch.Tensor:
    rot_rel = rot_pred.transpose(-1, -2) @ rot_gt
    trace = rot_rel[..., 0, 0] + rot_rel[..., 1, 1] + rot_rel[..., 2, 2]
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def pose_costs_and_residual_targets(
    candidate_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    rot_cost_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pose costs and left-update residual targets for candidate w2c poses."""
    if candidate_pose.ndim != 4 or candidate_pose.shape[-2:] != (4, 4):
        raise ValueError("candidate_pose must have shape (B,K,4,4)")
    if pose_gt.ndim != 3 or pose_gt.shape[-2:] != (4, 4):
        raise ValueError("pose_gt must have shape (B,4,4)")
    bsz, num_candidates = candidate_pose.shape[:2]
    if pose_gt.shape[0] != bsz:
        raise ValueError("pose_gt batch size must match candidate_pose")
    cand = candidate_pose.float()
    gt = pose_gt.float()
    gt_bank = gt[:, None].expand(-1, num_candidates, -1, -1)
    cand_centers = _camera_centers_from_w2c(cand)
    gt_centers = _camera_centers_from_w2c(gt)[:, None]
    trans_err = torch.linalg.norm(cand_centers - gt_centers, dim=-1)
    rot_err = _rotation_error_rad(cand[..., :3, :3], gt_bank[..., :3, :3])
    cost = trans_err + float(rot_cost_weight) * rot_err
    relative = gt_bank.reshape(bsz * num_candidates, 4, 4) @ pose_inverse(
        cand.reshape(bsz * num_candidates, 4, 4)
    )
    residual = se3_log(relative).reshape(bsz, num_candidates, 6)
    if valid_mask is not None:
        valid = valid_mask.to(device=cost.device).bool()
        if valid.shape != (bsz, num_candidates):
            raise ValueError(f"valid_mask must have shape {(bsz, num_candidates)}")
        cost = cost.masked_fill(~valid, float("inf"))
        residual = residual.masked_fill(~valid[..., None], 0.0)
        trans_err = trans_err.masked_fill(~valid, float("inf"))
        rot_err = rot_err.masked_fill(~valid, float("inf"))
    return cost, residual, trans_err, rot_err


def pose_energy_losses(
    outputs: Dict[str, torch.Tensor],
    candidate_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    target_temperature_m: float = 0.05,
    rot_cost_weight: float = 0.1,
    hard_ce_weight: float = 0.0,
    residual_weight: float = 1.0,
    improve_weight: float = 0.0,
    improve_margin_m: float = 0.0,
    update_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    energy_logits = outputs["energy_logits"].float()
    residual_delta = outputs["residual_delta"].float()
    if energy_logits.ndim != 2:
        raise ValueError("energy_logits must have shape (B,K)")
    if residual_delta.shape != (*energy_logits.shape, 6):
        raise ValueError("residual_delta must have shape (B,K,6)")
    bsz, num_candidates = energy_logits.shape
    valid = torch.ones((bsz, num_candidates), device=energy_logits.device, dtype=torch.bool)
    if valid_mask is not None:
        valid = valid_mask.to(device=energy_logits.device).bool()
    pose_cost, residual_target, trans_err, rot_err = pose_costs_and_residual_targets(
        candidate_pose.to(device=energy_logits.device),
        pose_gt.to(device=energy_logits.device),
        valid_mask=valid,
        rot_cost_weight=rot_cost_weight,
    )
    logits = energy_logits.masked_fill(~valid, -1.0e6)
    target_index = pose_cost.masked_fill(~valid, float("inf")).argmin(dim=1)
    target_logits = (-pose_cost / max(float(target_temperature_m), 1e-6)).masked_fill(~valid, -1.0e6)
    target_probs = F.softmax(target_logits, dim=1).detach()
    energy_loss = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    hard_ce_loss = F.cross_entropy(logits, target_index)

    residual_per = F.smooth_l1_loss(residual_delta, residual_target.to(residual_delta.dtype), reduction="none").mean(dim=-1)
    residual_loss = (residual_per * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    improve_loss = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if float(improve_weight) > 0.0:
        flat_pose = candidate_pose.to(device=energy_logits.device).reshape(bsz * num_candidates, 4, 4).float()
        flat_delta = (residual_delta.reshape(bsz * num_candidates, 6).float() * float(update_scale))
        updated = (se3_exp(flat_delta) @ flat_pose).reshape(bsz, num_candidates, 4, 4)
        updated_cost, _, _, _ = pose_costs_and_residual_targets(
            updated,
            pose_gt.to(device=energy_logits.device).float(),
            valid_mask=valid,
            rot_cost_weight=rot_cost_weight,
        )
        margin = float(improve_margin_m)
        improve_per = F.relu(updated_cost - pose_cost + margin)
        improve_loss = (improve_per * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    loss = (
        energy_loss
        + float(hard_ce_weight) * hard_ce_loss
        + float(residual_weight) * residual_loss
        + float(improve_weight) * improve_loss
    )
    pred_index = logits.argmax(dim=1)
    pred_cost = pose_cost.gather(1, pred_index[:, None]).squeeze(1)
    oracle_cost = pose_cost.gather(1, target_index[:, None]).squeeze(1)
    return {
        "loss": loss,
        "energy_loss": energy_loss,
        "hard_ce_loss": hard_ce_loss,
        "residual_loss": residual_loss,
        "improve_loss": improve_loss,
        "target_probs": target_probs,
        "pose_cost": pose_cost,
        "trans_err_m": trans_err,
        "rot_err_rad": rot_err,
        "target_index": target_index,
        "pred_index": pred_index,
        "pred_cost": pred_cost,
        "oracle_cost": oracle_cost,
        "top1_acc": (pred_index == target_index).float().mean(),
    }
