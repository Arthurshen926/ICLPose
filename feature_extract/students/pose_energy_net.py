"""Pose-conditioned neural energy and residual heads for CPR."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_refine.utils.lie_algebra import pose_inverse, se3_exp, se3_log


class PoseFeatureAdapter(nn.Module):
    """Residual adapter that turns anchored base features into localization features."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        residual_scale: float = 0.1,
        zero_init: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.l2_normalize = bool(l2_normalize)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1),
        )
        if bool(zero_init):
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("feature must have shape (B,C,H,W)")
        if feature.shape[1] != self.channels:
            raise ValueError(f"expected channels={self.channels}, got {feature.shape[1]}")
        adapted = feature + self.net(feature) * self.residual_scale
        if self.l2_normalize:
            adapted = F.normalize(adapted.float(), dim=1, eps=1.0e-6).to(dtype=adapted.dtype)
        return adapted


class PoseFeatureDomainAdapter(nn.Module):
    """Asymmetric query/map localization adapters with a shared interface."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        residual_scale: float = 0.1,
        zero_init: bool = True,
        l2_normalize: bool = True,
        uncertainty_enabled: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.query_adapter = PoseFeatureAdapter(
            channels=channels,
            hidden_dim=hidden_dim,
            residual_scale=residual_scale,
            zero_init=zero_init,
            l2_normalize=l2_normalize,
        )
        self.render_adapter = PoseFeatureAdapter(
            channels=channels,
            hidden_dim=hidden_dim,
            residual_scale=residual_scale,
            zero_init=zero_init,
            l2_normalize=l2_normalize,
        )
        if self.uncertainty_enabled:
            self.query_uncertainty = nn.Sequential(
                nn.Conv2d(self.channels, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
            self.render_uncertainty = nn.Sequential(
                nn.Conv2d(self.channels, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
        else:
            self.query_uncertainty = None
            self.render_uncertainty = None

    def project_query(self, feature: torch.Tensor) -> torch.Tensor:
        return self.query_adapter(feature)

    def project_render(self, feature: torch.Tensor) -> torch.Tensor:
        return self.render_adapter(feature)

    def _default_uncertainty(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (feature.shape[0], 1, feature.shape[-2], feature.shape[-1]),
            device=feature.device,
            dtype=feature.dtype,
        )

    def project_query_with_uncertainty(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.project_query(feature)
        if self.query_uncertainty is None:
            return loc, self._default_uncertainty(feature)
        return loc, torch.sigmoid(self.query_uncertainty(feature.float())).to(dtype=loc.dtype)

    def project_render_with_uncertainty(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.project_render(feature)
        if self.render_uncertainty is None:
            return loc, self._default_uncertainty(feature)
        return loc, torch.sigmoid(self.render_uncertainty(feature.float())).to(dtype=loc.dtype)

    def forward(self, feature: torch.Tensor, domain: str = "query") -> torch.Tensor:
        domain = str(domain).lower()
        if domain == "query":
            return self.project_query(feature)
        if domain in {"render", "map"}:
            return self.project_render(feature)
        raise ValueError(f"Unknown pose feature adapter domain: {domain}")


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
        zero_init_residual_head: bool = False,
        factorized_heads: bool = False,
    ):
        super().__init__()
        self.vector_dim = int(vector_dim)
        self.score_map_channels = int(score_map_channels)
        self.map_channels = int(map_channels)
        self.grid_size = int(grid_size)
        self.hidden_dim = int(hidden_dim)
        self.context_layers = int(context_layers)
        self.context_heads = int(context_heads)
        self.factorized_heads = bool(factorized_heads)
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
        if self.factorized_heads:
            self.translation_energy_head = nn.Linear(self.hidden_dim, 1)
            self.rotation_energy_head = nn.Linear(self.hidden_dim, 1)
        else:
            self.translation_energy_head = None
            self.rotation_energy_head = None
        if bool(zero_init_heads):
            heads = [self.energy_head, self.residual_head, self.confidence_head]
            if self.translation_energy_head is not None:
                heads.append(self.translation_energy_head)
            if self.rotation_energy_head is not None:
                heads.append(self.rotation_energy_head)
            for head in heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        elif bool(zero_init_residual_head):
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

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
        translation_energy_logits = None
        rotation_energy_logits = None
        if self.factorized_heads:
            translation_energy_logits = self.translation_energy_head(tokens).squeeze(-1).float()
            rotation_energy_logits = self.rotation_energy_head(tokens).squeeze(-1).float()
        if valid_mask is not None:
            valid = valid_mask.to(device=energy_logits.device).bool()
            if valid.shape != (bsz, num_candidates):
                raise ValueError(f"valid_mask must have shape {(bsz, num_candidates)}")
            energy_logits = energy_logits.masked_fill(~valid, -1.0e6)
            confidence_logits = confidence_logits.masked_fill(~valid, -1.0e6)
            residual_delta = residual_delta.masked_fill(~valid[..., None], 0.0)
            if translation_energy_logits is not None:
                translation_energy_logits = translation_energy_logits.masked_fill(~valid, -1.0e6)
            if rotation_energy_logits is not None:
                rotation_energy_logits = rotation_energy_logits.masked_fill(~valid, -1.0e6)
        result = {
            "energy_logits": energy_logits,
            "residual_delta": residual_delta,
            "confidence_logits": confidence_logits,
            "tokens": tokens,
        }
        if self.factorized_heads:
            result["translation_energy_logits"] = translation_energy_logits
            result["rotation_energy_logits"] = rotation_energy_logits
            result["joint_energy_logits"] = energy_logits
        return result


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


def camera_centers_from_w2c_pose(pose_w2c: torch.Tensor) -> torch.Tensor:
    """Return camera centers for one or more world-to-camera poses."""
    pose = pose_w2c.float()
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)


def compose_w2c_from_center_and_rotation(center_pose_w2c: torch.Tensor, rotation_pose_w2c: torch.Tensor) -> torch.Tensor:
    """Compose a w2c pose from the camera center of one pose and rotation of another."""
    center_pose = center_pose_w2c.float()
    rotation_pose = rotation_pose_w2c.float()
    if center_pose.shape[-2:] != (4, 4) or rotation_pose.shape[-2:] != (4, 4):
        raise ValueError("poses must end with shape (4,4)")
    center_pose, rotation_pose = torch.broadcast_tensors(center_pose, rotation_pose)
    center = camera_centers_from_w2c_pose(center_pose)
    rotation = rotation_pose[..., :3, :3]
    composed = rotation_pose.clone()
    composed[..., :3, 3] = -(rotation @ center.unsqueeze(-1)).squeeze(-1)
    composed[..., 3, :] = center_pose.new_tensor([0.0, 0.0, 0.0, 1.0]).expand_as(composed[..., 3, :])
    dtype = torch.promote_types(center_pose_w2c.dtype, rotation_pose_w2c.dtype)
    return composed.to(dtype=dtype)


def pose_energy_factorized_selection(
    outputs: Dict[str, torch.Tensor],
    candidate_pose: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select center and rotation with factorized heads, then compose a final pose."""
    if candidate_pose.ndim != 4 or candidate_pose.shape[-2:] != (4, 4):
        raise ValueError("candidate_pose must have shape (B,K,4,4)")
    energy_logits = outputs["energy_logits"].float()
    if energy_logits.ndim != 2:
        raise ValueError("energy_logits must have shape (B,K)")
    bsz, num_candidates = energy_logits.shape
    if candidate_pose.shape[:2] != (bsz, num_candidates):
        raise ValueError(
            f"candidate_pose shape {tuple(candidate_pose.shape[:2])} does not match logits {(bsz, num_candidates)}"
        )
    valid = torch.ones((bsz, num_candidates), device=energy_logits.device, dtype=torch.bool)
    if valid_mask is not None:
        valid = valid_mask.to(device=energy_logits.device).bool()
        if valid.shape != (bsz, num_candidates):
            raise ValueError(f"valid_mask must have shape {(bsz, num_candidates)}")
    trans_logits = outputs.get("translation_energy_logits", energy_logits).float().masked_fill(~valid, -1.0e6)
    rot_logits = outputs.get("rotation_energy_logits", energy_logits).float().masked_fill(~valid, -1.0e6)
    trans_idx = trans_logits.argmax(dim=1)
    rot_idx = rot_logits.argmax(dim=1)
    batch_idx = torch.arange(bsz, device=energy_logits.device)
    candidate_pose = candidate_pose.to(device=energy_logits.device)
    center_pose = candidate_pose[batch_idx, trans_idx]
    rotation_pose = candidate_pose[batch_idx, rot_idx]
    selected_pose = compose_w2c_from_center_and_rotation(center_pose, rotation_pose)
    return selected_pose.to(dtype=candidate_pose.dtype), trans_idx, rot_idx


def pose_energy_selection_scores(
    outputs: Dict[str, torch.Tensor],
    *,
    confidence_weight: float = 0.0,
    residual_norm_weight: float = 0.0,
    residual_trans_scale_m: float = 1.0,
    residual_rot_scale_rad: float = 1.0,
) -> torch.Tensor:
    """Build inference selection logits from energy, confidence, and residual size."""
    scores = outputs["energy_logits"].float()
    if float(confidence_weight) != 0.0:
        confidence_logits = outputs.get("confidence_logits")
        if confidence_logits is None:
            raise ValueError("confidence_weight requires confidence_logits")
        if confidence_logits.shape != scores.shape:
            raise ValueError(f"confidence_logits must have shape {tuple(scores.shape)}")
        scores = scores + float(confidence_weight) * confidence_logits.float()
    if float(residual_norm_weight) != 0.0:
        residual_delta = outputs.get("residual_delta")
        if residual_delta is None:
            raise ValueError("residual_norm_weight requires residual_delta")
        if residual_delta.shape != (*scores.shape, 6):
            raise ValueError(f"residual_delta must have shape {tuple((*scores.shape, 6))}")
        residual_scale = residual_delta.new_tensor(
            [
                max(float(residual_trans_scale_m), 1.0e-6),
                max(float(residual_trans_scale_m), 1.0e-6),
                max(float(residual_trans_scale_m), 1.0e-6),
                max(float(residual_rot_scale_rad), 1.0e-6),
                max(float(residual_rot_scale_rad), 1.0e-6),
                max(float(residual_rot_scale_rad), 1.0e-6),
            ]
        )
        residual_norm = torch.linalg.norm(residual_delta.float() / residual_scale, dim=-1)
        scores = scores - float(residual_norm_weight) * residual_norm
    return scores


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
    residual_target_mode: str = "all",
    residual_soft_topk_temperature_m: float = 0.05,
    improve_weight: float = 0.0,
    improve_margin_m: float = 0.0,
    update_scale: float = 1.0,
    residual_trans_scale_m: float = 1.0,
    residual_rot_scale_rad: float = 1.0,
    anti_identity_weight: float = 0.0,
    anti_identity_margin: float = 0.0,
    anti_identity_min_gap_m: float = 0.03,
    identity_index: int = 0,
    pairwise_rank_weight: float = 0.0,
    pairwise_rank_min_gap_m: float = 0.03,
    pairwise_rank_logit_margin: float = 0.5,
    component_hard_ce_weight: float = 0.0,
    translation_energy_weight: float = 0.0,
    rotation_energy_weight: float = 0.0,
    joint_energy_weight: float = 0.0,
    confidence_weight: float = 0.0,
    confidence_temperature_m: float | None = None,
) -> Dict[str, torch.Tensor]:
    energy_logits = outputs["energy_logits"].float()
    residual_delta = outputs["residual_delta"].float()
    confidence_logits = outputs.get("confidence_logits")
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

    def _component_energy_loss(head_name: str, cost: torch.Tensor) -> torch.Tensor:
        component_logits = outputs.get(head_name)
        if component_logits is None:
            return torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
        component_logits = component_logits.float()
        if component_logits.shape != (bsz, num_candidates):
            raise ValueError(f"{head_name} must have shape {(bsz, num_candidates)}")
        component_logits = component_logits.masked_fill(~valid, -1.0e6)
        component_target = (-cost / max(float(target_temperature_m), 1e-6)).masked_fill(~valid, -1.0e6)
        component_probs = F.softmax(component_target, dim=1).detach()
        return -(component_probs * F.log_softmax(component_logits, dim=1)).sum(dim=1).mean()

    translation_energy_loss = _component_energy_loss("translation_energy_logits", trans_err)
    rotation_energy_loss = _component_energy_loss("rotation_energy_logits", rot_err * float(rot_cost_weight))
    joint_energy_loss = _component_energy_loss("joint_energy_logits", pose_cost)
    component_hard_ce_loss = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if float(component_hard_ce_weight) > 0.0:
        trans_logits = outputs.get("translation_energy_logits")
        rot_logits = outputs.get("rotation_energy_logits")
        if trans_logits is None or rot_logits is None:
            raise ValueError("component_hard_ce_weight > 0 requires translation/rotation energy heads")
        trans_logits = trans_logits.float().masked_fill(~valid, -1.0e6)
        rot_logits = rot_logits.float().masked_fill(~valid, -1.0e6)
        trans_target_index = trans_err.masked_fill(~valid, float("inf")).argmin(dim=1)
        rot_target_index = rot_err.masked_fill(~valid, float("inf")).argmin(dim=1)
        component_hard_ce_loss = 0.5 * (
            F.cross_entropy(trans_logits, trans_target_index)
            + F.cross_entropy(rot_logits, rot_target_index)
        )
    confidence_loss = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if float(confidence_weight) > 0.0:
        if confidence_logits is None:
            raise ValueError("confidence_weight > 0 requires confidence_logits")
        confidence_logits = confidence_logits.float()
        if confidence_logits.shape != (bsz, num_candidates):
            raise ValueError(f"confidence_logits must have shape {(bsz, num_candidates)}")
        conf_temp = max(float(confidence_temperature_m or target_temperature_m), 1.0e-6)
        confidence_target = torch.exp(-pose_cost / conf_temp).detach()
        confidence_target = torch.where(valid, confidence_target, torch.zeros_like(confidence_target))
        confidence_per = F.binary_cross_entropy_with_logits(
            confidence_logits,
            confidence_target,
            reduction="none",
        )
        confidence_loss = (confidence_per * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    residual_scale = residual_delta.new_tensor(
        [
            max(float(residual_trans_scale_m), 1.0e-6),
            max(float(residual_trans_scale_m), 1.0e-6),
            max(float(residual_trans_scale_m), 1.0e-6),
            max(float(residual_rot_scale_rad), 1.0e-6),
            max(float(residual_rot_scale_rad), 1.0e-6),
            max(float(residual_rot_scale_rad), 1.0e-6),
        ]
    )
    residual_per = F.smooth_l1_loss(
        residual_delta / residual_scale,
        residual_target.to(residual_delta.dtype) / residual_scale,
        reduction="none",
    ).mean(dim=-1)
    residual_mode = str(residual_target_mode or "all").lower()
    if residual_mode == "all":
        residual_weights = valid.float()
    elif residual_mode in {"top1", "best"}:
        residual_weights = torch.zeros_like(residual_per)
        residual_weights.scatter_(1, target_index[:, None], 1.0)
        residual_weights = residual_weights * valid.float()
    elif residual_mode in {"soft_topk", "soft"}:
        residual_weights = F.softmax(
            (-pose_cost / max(float(residual_soft_topk_temperature_m), 1.0e-6)).masked_fill(~valid, -1.0e6),
            dim=1,
        ).detach()
        residual_weights = residual_weights * valid.float()
    else:
        raise ValueError(f"Unknown residual_target_mode: {residual_target_mode}")
    residual_loss = (residual_per * residual_weights).sum() / residual_weights.sum().clamp(min=1.0)

    anti_identity_loss = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    anti_identity_active = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if float(anti_identity_weight) > 0.0 and 0 <= int(identity_index) < num_candidates:
        id_idx = int(identity_index)
        id_cost = pose_cost[:, id_idx : id_idx + 1]
        id_logit = logits[:, id_idx : id_idx + 1]
        better_than_identity = (
            valid
            & torch.isfinite(pose_cost)
            & torch.isfinite(id_cost)
            & (pose_cost + float(anti_identity_min_gap_m) < id_cost)
        )
        anti_identity_active = better_than_identity.float().amax(dim=1).mean()
        if bool(better_than_identity.any()):
            anti_per = F.relu(id_logit - logits + float(anti_identity_margin))
            anti_identity_loss = anti_per[better_than_identity].mean()

    pairwise_rank_loss = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    pairwise_rank_active = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if float(pairwise_rank_weight) > 0.0:
        best_cost = pose_cost.gather(1, target_index[:, None])
        best_logit = logits.gather(1, target_index[:, None])
        worse_than_best = (
            valid
            & torch.isfinite(pose_cost)
            & (pose_cost > best_cost + float(pairwise_rank_min_gap_m))
        )
        pairwise_rank_active = worse_than_best.float().mean()
        if bool(worse_than_best.any()):
            pairwise_per = F.softplus(logits - best_logit + float(pairwise_rank_logit_margin))
            pairwise_rank_loss = pairwise_per[worse_than_best].mean()

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
        + float(anti_identity_weight) * anti_identity_loss
        + float(pairwise_rank_weight) * pairwise_rank_loss
        + float(component_hard_ce_weight) * component_hard_ce_loss
        + float(improve_weight) * improve_loss
        + float(translation_energy_weight) * translation_energy_loss
        + float(rotation_energy_weight) * rotation_energy_loss
        + float(joint_energy_weight) * joint_energy_loss
        + float(confidence_weight) * confidence_loss
    )
    pred_index = logits.argmax(dim=1)
    pred_cost = pose_cost.gather(1, pred_index[:, None]).squeeze(1)
    oracle_cost = pose_cost.gather(1, target_index[:, None]).squeeze(1)
    return {
        "loss": loss,
        "energy_loss": energy_loss,
        "hard_ce_loss": hard_ce_loss,
        "translation_energy_loss": translation_energy_loss,
        "rotation_energy_loss": rotation_energy_loss,
        "joint_energy_loss": joint_energy_loss,
        "component_hard_ce_loss": component_hard_ce_loss,
        "confidence_loss": confidence_loss,
        "residual_loss": residual_loss,
        "anti_identity_loss": anti_identity_loss,
        "anti_identity_active": anti_identity_active,
        "pairwise_rank_loss": pairwise_rank_loss,
        "pairwise_rank_active": pairwise_rank_active,
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
