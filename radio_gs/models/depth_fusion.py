from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_feature_size(
    feat: torch.Tensor,
    fH: int,
    fW: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize a feature map to the probe resolution."""
    if feat.shape[-2:] != (fH, fW):
        feat = F.interpolate(
            feat.unsqueeze(0).to(device),
            (fH, fW),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    else:
        feat = feat.to(device)
    return feat.float()


def ensure_depth_size(
    depth: torch.Tensor | np.ndarray | None,
    fH: int,
    fW: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert a depth map to a float tensor at the probe resolution."""
    if depth is None:
        depth_t = torch.zeros((fH, fW), dtype=torch.float32, device=device)
    elif isinstance(depth, np.ndarray):
        depth_t = torch.from_numpy(depth.astype(np.float32)).to(device)
    else:
        depth_t = depth.to(device).float()

    if depth_t.dim() == 4:
        depth_t = depth_t.squeeze(0).squeeze(0)
    elif depth_t.dim() == 3:
        depth_t = depth_t.squeeze(0)

    if depth_t.shape != (fH, fW):
        depth_t = F.interpolate(
            depth_t.unsqueeze(0).unsqueeze(0),
            (fH, fW),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    return depth_t.float()


def ensure_alpha_size(
    alpha: torch.Tensor | np.ndarray | None,
    fH: int,
    fW: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert an alpha/confidence map to a float tensor at the probe resolution."""
    alpha_t = ensure_depth_size(alpha, fH, fW, device)
    return alpha_t.clamp_(0.0, 1.0)


def align_depth_scale_shift(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Least-squares align source depth to target depth using valid pixels."""
    if valid_mask.sum() < 10:
        return source_depth.float()

    src_vals = source_depth[valid_mask].float()
    tgt_vals = target_depth[valid_mask].float()
    design = torch.stack([src_vals, torch.ones_like(src_vals)], dim=1)
    params = torch.linalg.lstsq(design, tgt_vals).solution
    if not torch.isfinite(params).all():
        return source_depth.float()
    return source_depth.float() * params[0] + params[1]


def depth_gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    """Compute a simple gradient-magnitude confidence cue for depth maps."""
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[:, :-1] = depth[:, 1:] - depth[:, :-1]
    grad_y[:-1, :] = depth[1:, :] - depth[:-1, :]
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


@torch.no_grad()
def prepare_depth_fusion_sample(
    feat: torch.Tensor,
    geom_depth: torch.Tensor | np.ndarray | None,
    geom_alpha: torch.Tensor | np.ndarray | None,
    depth_probe: nn.Module,
    fH: int,
    fW: int,
    device: torch.device,
    output_device: Optional[Union[torch.device, str]] = None,
) -> Dict[str, torch.Tensor]:
    """Build feature-aligned inputs for learned depth fusion."""
    sample_device = torch.device(output_device) if output_device is not None else device
    feat_t = ensure_feature_size(feat, fH, fW, device)
    C = feat_t.shape[0]

    feat_depth = depth_probe(feat_t.reshape(C, -1).T).squeeze(-1).reshape(fH, fW).float()

    geom_t = ensure_depth_size(geom_depth, fH, fW, device)
    geom_valid = (geom_t > 0.01).float()
    geom_alpha_t = ensure_alpha_size(geom_alpha, fH, fW, device) if geom_alpha is not None else geom_valid.clone()
    geom_alpha_t = geom_alpha_t * geom_valid
    align_mask = geom_valid > 0.5
    # Align feature depth to geometric depth (geom is metric and more accurate)
    feat_depth = align_depth_scale_shift(feat_depth, geom_t, align_mask)
    geom_aligned = geom_t

    diff = geom_aligned - feat_depth
    abs_diff = diff.abs()
    rel_diff = diff / feat_depth.abs().clamp(min=0.1)
    feat_grad = depth_gradient_magnitude(feat_depth)
    geom_grad = depth_gradient_magnitude(geom_aligned) * geom_valid
    grad_diff = (geom_grad - feat_grad).abs()
    feat_norm = feat_t.square().mean(dim=0).sqrt()
    alpha_grad = depth_gradient_magnitude(geom_alpha_t)

    feat_flat = feat_t.reshape(C, -1).T
    feat_depth_flat = feat_depth.reshape(-1, 1)
    geom_flat = geom_aligned.reshape(-1, 1)
    geom_valid_flat = geom_valid.reshape(-1, 1)
    geom_alpha_flat = geom_alpha_t.reshape(-1, 1)

    extras = torch.cat(
        [
            feat_depth_flat,
            geom_flat,
            geom_valid_flat,
            geom_alpha_flat,
            diff.reshape(-1, 1),
            abs_diff.reshape(-1, 1),
            rel_diff.reshape(-1, 1),
            feat_grad.reshape(-1, 1),
            geom_grad.reshape(-1, 1),
            grad_diff.reshape(-1, 1),
            feat_norm.reshape(-1, 1),
            (1.0 - geom_alpha_flat),
            alpha_grad.reshape(-1, 1),
        ],
        dim=1,
    )

    def _move(t: torch.Tensor) -> torch.Tensor:
        return t.to(sample_device, non_blocking=True) if t.device != sample_device else t

    return {
        "input_flat": _move(torch.cat([feat_flat, extras], dim=1)),
        "feat_depth_flat": _move(feat_depth_flat),
        "geom_depth_flat": _move(geom_flat),
        "geom_valid_flat": _move(geom_valid_flat),
        "geom_alpha_flat": _move(geom_alpha_flat),
        "feat_depth_map": _move(feat_depth),
        "geom_depth_map": _move(geom_aligned),
        "geom_valid_map": _move(geom_valid),
        "geom_alpha_map": _move(geom_alpha_t),
    }


class DepthFusionProbe(nn.Module):
    """Blend feature-predicted and geometric depth with learned gating."""

    def __init__(self, in_dim: int, hidden: int = 256, geom_bias: float = 4.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden, 1)
        self.residual_head = nn.Linear(hidden, 1)
        # Initialize gate bias to strongly favor geometric depth (sigmoid(4.0) ≈ 0.98)
        nn.init.constant_(self.gate_head.bias, geom_bias)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        input_flat: torch.Tensor,
        feat_depth_flat: torch.Tensor,
        geom_depth_flat: torch.Tensor,
        geom_valid_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_flat)
        gate = torch.sigmoid(self.gate_head(hidden)) * geom_valid_flat
        base = gate * geom_depth_flat + (1.0 - gate) * feat_depth_flat
        correction_scale = (geom_depth_flat - feat_depth_flat).abs().clamp(max=0.3) + 0.05
        residual = torch.tanh(self.residual_head(hidden)) * correction_scale
        return base + residual, gate


def _compute_gate_target(
    feat_depth_flat: torch.Tensor,
    geom_depth_flat: torch.Tensor,
    targets: torch.Tensor,
    geom_valid_flat: torch.Tensor,
) -> torch.Tensor:
    """Soft supervision target for geometry-reliability gating."""
    feat_err = (feat_depth_flat.squeeze(-1) - targets).abs()
    geom_err = (geom_depth_flat.squeeze(-1) - targets).abs()
    gate_target = feat_err / (feat_err + geom_err + 1e-6)
    gate_target = gate_target.unsqueeze(-1)
    return gate_target * geom_valid_flat


def sample_depth_fusion_training_pixels(
    sample: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    max_samples: Optional[int] = None,
    valid_threshold: float = 0.01,
    min_valid: int = 10,
    generator: Optional[torch.Generator] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Subsample valid fusion-training pixels to keep memory bounded."""
    target_t = targets.float()
    if target_t.dim() == 4:
        target_t = target_t.squeeze(0).squeeze(0)
    elif target_t.dim() == 3:
        target_t = target_t.squeeze(0)

    valid_idx = (target_t > valid_threshold).reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() < min_valid:
        return None

    if max_samples is not None and max_samples > 0 and valid_idx.numel() > max_samples:
        perm = torch.randperm(valid_idx.numel(), generator=generator)
        valid_idx = valid_idx[perm[:max_samples]]

    return {
        "input_flat": sample["input_flat"].index_select(0, valid_idx),
        "feat_depth_flat": sample["feat_depth_flat"].index_select(0, valid_idx),
        "geom_depth_flat": sample["geom_depth_flat"].index_select(0, valid_idx),
        "geom_valid_flat": sample["geom_valid_flat"].index_select(0, valid_idx),
        "targets": target_t.reshape(-1).index_select(0, valid_idx),
    }


def train_depth_fusion_probe(
    train_input_flat: torch.Tensor,
    train_feat_depth_flat: torch.Tensor,
    train_geom_depth_flat: torch.Tensor,
    train_geom_valid_flat: torch.Tensor,
    train_targets: torch.Tensor,
    device: torch.device,
    *,
    epochs: int = 500,
    batch_size: int = 16384,
    lr: float = 5e-4,
    reliability_weight: float = 0.1,
) -> DepthFusionProbe:
    """Train the learned depth-fusion probe."""
    probe = DepthFusionProbe(train_input_flat.shape[1]).to(device).train()
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_targets.shape[0]
    storage_device = train_targets.device

    for _ in range(epochs):
        idx = torch.randint(0, n, (min(batch_size, n),), device=storage_device)
        batch_input = train_input_flat[idx].to(device, non_blocking=True)
        batch_feat_depth = train_feat_depth_flat[idx].to(device, non_blocking=True)
        batch_geom_depth = train_geom_depth_flat[idx].to(device, non_blocking=True)
        batch_geom_valid = train_geom_valid_flat[idx].to(device, non_blocking=True)
        batch_targets = train_targets[idx].to(device, non_blocking=True)
        pred, gate = probe(
            batch_input,
            batch_feat_depth,
            batch_geom_depth,
            batch_geom_valid,
        )
        loss = F.smooth_l1_loss(pred.squeeze(-1), batch_targets)
        if reliability_weight > 0:
            gate_target = _compute_gate_target(
                batch_feat_depth,
                batch_geom_depth,
                batch_targets,
                batch_geom_valid,
            )
            gate_loss = F.smooth_l1_loss(gate, gate_target)
            loss = loss + reliability_weight * gate_loss
        loss = loss + 0.01 * (gate * (1.0 - gate)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    probe.eval()
    return probe


@torch.no_grad()
def predict_depth_fusion(
    probe: DepthFusionProbe,
    sample: Dict[str, torch.Tensor],
    fH: int,
    fW: int,
) -> Dict[str, Any]:
    """Run the learned depth-fusion probe and return depth and gate maps."""
    pred, gate = probe(
        sample["input_flat"],
        sample["feat_depth_flat"],
        sample["geom_depth_flat"],
        sample["geom_valid_flat"],
    )
    return {
        "depth": pred.squeeze(-1).reshape(fH, fW),
        "gate": gate.squeeze(-1).reshape(fH, fW),
        "reliability": gate.squeeze(-1).reshape(fH, fW),
        "feat_depth": sample["feat_depth_map"],
        "geom_depth": sample["geom_depth_map"],
    }
