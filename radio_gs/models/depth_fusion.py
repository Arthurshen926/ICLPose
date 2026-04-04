from __future__ import annotations

from typing import Any, Dict

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
    depth_probe: nn.Module,
    fH: int,
    fW: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build feature-aligned inputs for learned depth fusion."""
    feat_t = ensure_feature_size(feat, fH, fW, device)
    C = feat_t.shape[0]

    feat_depth = depth_probe(feat_t.reshape(C, -1).T).squeeze(-1).reshape(fH, fW).float()

    geom_t = ensure_depth_size(geom_depth, fH, fW, device)
    geom_valid = (geom_t > 0.01).float()
    align_mask = geom_valid > 0.5
    geom_aligned = align_depth_scale_shift(geom_t, feat_depth, align_mask)
    geom_aligned = torch.where(align_mask, geom_aligned, feat_depth)

    diff = geom_aligned - feat_depth
    abs_diff = diff.abs()
    rel_diff = diff / feat_depth.abs().clamp(min=0.1)
    feat_grad = depth_gradient_magnitude(feat_depth)
    geom_grad = depth_gradient_magnitude(geom_aligned) * geom_valid
    grad_diff = (geom_grad - feat_grad).abs()
    feat_norm = feat_t.square().mean(dim=0).sqrt()

    feat_flat = feat_t.reshape(C, -1).T
    feat_depth_flat = feat_depth.reshape(-1, 1)
    geom_flat = geom_aligned.reshape(-1, 1)
    geom_valid_flat = geom_valid.reshape(-1, 1)

    extras = torch.cat(
        [
            feat_depth_flat,
            geom_flat,
            geom_valid_flat,
            diff.reshape(-1, 1),
            abs_diff.reshape(-1, 1),
            rel_diff.reshape(-1, 1),
            feat_grad.reshape(-1, 1),
            geom_grad.reshape(-1, 1),
            grad_diff.reshape(-1, 1),
            feat_norm.reshape(-1, 1),
        ],
        dim=1,
    )

    return {
        "input_flat": torch.cat([feat_flat, extras], dim=1),
        "feat_depth_flat": feat_depth_flat,
        "geom_depth_flat": geom_flat,
        "geom_valid_flat": geom_valid_flat,
        "feat_depth_map": feat_depth,
        "geom_depth_map": geom_aligned,
        "geom_valid_map": geom_valid,
    }


class DepthFusionProbe(nn.Module):
    """Blend feature-predicted and geometric depth with learned gating."""

    def __init__(self, in_dim: int, hidden: int = 256, geom_bias: float = 2.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden, 1)
        self.residual_head = nn.Linear(hidden, 1)
        # Initialize gate bias to favor geometric depth (sigmoid(2.0) ≈ 0.88)
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
        correction_scale = (geom_depth_flat - feat_depth_flat).abs().clamp(max=0.5) + 0.1
        residual = torch.tanh(self.residual_head(hidden)) * correction_scale
        return base + residual, gate


def train_depth_fusion_probe(
    train_input_flat: torch.Tensor,
    train_feat_depth_flat: torch.Tensor,
    train_geom_depth_flat: torch.Tensor,
    train_geom_valid_flat: torch.Tensor,
    train_targets: torch.Tensor,
    device: torch.device,
    *,
    epochs: int = 300,
    batch_size: int = 16384,
    lr: float = 1e-3,
) -> DepthFusionProbe:
    """Train the learned depth-fusion probe."""
    probe = DepthFusionProbe(train_input_flat.shape[1]).to(device).train()
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_targets.shape[0]

    for _ in range(epochs):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        pred, gate = probe(
            train_input_flat[idx],
            train_feat_depth_flat[idx],
            train_geom_depth_flat[idx],
            train_geom_valid_flat[idx],
        )
        loss = F.smooth_l1_loss(pred.squeeze(-1), train_targets[idx])
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
        "feat_depth": sample["feat_depth_map"],
        "geom_depth": sample["geom_depth_map"],
    }
