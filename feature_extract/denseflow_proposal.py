from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _intrinsics_components(intrinsics: torch.Tensor, batch_size: int, device, dtype):
    intr = intrinsics.to(device=device, dtype=dtype)
    if intr.ndim == 1:
        intr = intr.view(1, 4)
    if intr.ndim != 2 or intr.shape[1] != 4:
        raise ValueError("intrinsics must have shape (4,) or (B,4)")
    if intr.shape[0] == 1 and batch_size > 1:
        intr = intr.expand(batch_size, -1)
    if intr.shape[0] != batch_size:
        raise ValueError(f"intrinsics batch {intr.shape[0]} does not match B={batch_size}")
    return intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]


def scale_intrinsics_for_hw(
    intrinsics: torch.Tensor,
    *,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    src_h, src_w = int(source_hw[0]), int(source_hw[1])
    dst_h, dst_w = int(target_hw[0]), int(target_hw[1])
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    scaled = intrinsics.clone()
    scaled[..., 0] = scaled[..., 0] * sx
    scaled[..., 1] = scaled[..., 1] * sy
    scaled[..., 2] = scaled[..., 2] * sx
    scaled[..., 3] = scaled[..., 3] * sy
    return scaled


def denseflow_gt_flow_from_render_depth(
    *,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    render_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    if pose_init.ndim != 3 or pose_init.shape[-2:] != (4, 4):
        raise ValueError("pose_init must have shape (B,4,4)")
    if pose_gt.shape != pose_init.shape:
        raise ValueError("pose_gt must have shape (B,4,4)")
    depth = render_depth.float()
    if depth.ndim == 3:
        depth = depth[:, None]
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("render_depth must have shape (B,1,H,W) or (B,H,W)")
    if target_hw is not None and tuple(depth.shape[-2:]) != tuple(target_hw):
        depth = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=False)
    bsz, _c, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype
    fx, fy, cx, cy = _intrinsics_components(intrinsics, bsz, device, dtype)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, height, width).expand(bsz, -1, -1, -1)
    yy = yy.view(1, 1, height, width).expand(bsz, -1, -1, -1)
    z = depth.clamp(min=1.0e-6)
    x = (xx - cx.view(bsz, 1, 1, 1)) * z / fx.view(bsz, 1, 1, 1).clamp(min=1.0e-6)
    y = (yy - cy.view(bsz, 1, 1, 1)) * z / fy.view(bsz, 1, 1, 1).clamp(min=1.0e-6)
    cam_init = torch.cat([x, y, z, torch.ones_like(z)], dim=1).flatten(2)
    rel = torch.bmm(pose_gt.float(), torch.linalg.inv(pose_init.float())).to(device=device, dtype=dtype)
    cam_gt = torch.bmm(rel[:, :3], cam_init).view(bsz, 3, height, width)
    z_gt = cam_gt[:, 2:3]
    u_gt = fx.view(bsz, 1, 1, 1) * cam_gt[:, 0:1] / z_gt.clamp(min=1.0e-6) + cx.view(
        bsz, 1, 1, 1
    )
    v_gt = fy.view(bsz, 1, 1, 1) * cam_gt[:, 1:2] / z_gt.clamp(min=1.0e-6) + cy.view(
        bsz, 1, 1, 1
    )
    flow = torch.cat([u_gt - xx, v_gt - yy], dim=1)
    valid = (
        (depth > 0.05)
        & (z_gt > 0.05)
        & (u_gt >= -0.5)
        & (u_gt <= float(width) - 0.5)
        & (v_gt >= -0.5)
        & (v_gt <= float(height) - 0.5)
    )
    return {"flow": flow * valid.to(dtype=flow.dtype), "valid": valid}


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(max(1, min(8, out_ch // 8)), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dense_local_correlation(query: torch.Tensor, render: torch.Tensor, radius: int) -> torch.Tensor:
    if query.shape != render.shape:
        raise ValueError(f"query and render must have same shape, got {query.shape} and {render.shape}")
    radius = int(radius)
    query_n = F.normalize(query.float(), dim=1, eps=1.0e-6)
    render_n = F.normalize(render.float(), dim=1, eps=1.0e-6)
    padded = F.pad(query_n, [radius] * 4)
    rows = []
    _, _, height, width = render_n.shape
    for dy in range(-radius, radius + 1):
        y0 = dy + radius
        for dx in range(-radius, radius + 1):
            x0 = dx + radius
            sample = padded[:, :, y0 : y0 + height, x0 : x0 + width]
            rows.append((render_n * sample).sum(dim=1))
    return torch.stack(rows, dim=1)


class PofdDenseFlowHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        radius: int = 8,
        hidden_dim: int = 64,
        zero_init: bool = True,
        max_flow_px: float | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.radius = int(radius)
        self.max_flow_px = float(max_flow_px if max_flow_px is not None else radius)
        corr_ch = (2 * self.radius + 1) ** 2
        context_ch = 5
        self.predict = nn.Sequential(
            ConvNormAct(corr_ch + context_ch, int(hidden_dim)),
            ConvNormAct(int(hidden_dim), int(hidden_dim)),
            nn.Conv2d(int(hidden_dim), 3, 1),
        )
        if zero_init:
            nn.init.zeros_(self.predict[-1].weight)
            if self.predict[-1].bias is not None:
                nn.init.zeros_(self.predict[-1].bias)

    def _context(self, depth: torch.Tensor | None, corr: torch.Tensor) -> torch.Tensor:
        bsz, _corr_ch, height, width = corr.shape
        device, dtype = corr.device, corr.dtype
        if depth is None:
            depth_ch = torch.zeros(bsz, 1, height, width, device=device, dtype=dtype)
            inv_depth = torch.zeros_like(depth_ch)
            valid = torch.ones_like(depth_ch)
        else:
            depth_ch = depth.float()
            if depth_ch.ndim == 3:
                depth_ch = depth_ch[:, None]
            if depth_ch.shape[-2:] != (height, width):
                depth_ch = F.interpolate(depth_ch, (height, width), mode="bilinear", align_corners=False)
            valid = (depth_ch > 0.05).to(dtype=dtype)
            log_depth = torch.log(depth_ch.clamp(min=0.05))
            inv_depth = 1.0 / depth_ch.clamp(min=0.05)
            depth_ch = (log_depth - log_depth.mean(dim=(-2, -1), keepdim=True)) / log_depth.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp(min=1.0e-6)
            inv_depth = (inv_depth - inv_depth.mean(dim=(-2, -1), keepdim=True)) / inv_depth.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp(min=1.0e-6)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=0).view(1, 2, height, width).expand(bsz, -1, -1, -1)
        return torch.cat([depth_ch.to(dtype=dtype), inv_depth.to(dtype=dtype), xy, valid], dim=1)

    def forward(
        self,
        query: torch.Tensor,
        render: torch.Tensor,
        *,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del intrinsics
        if query.ndim != 4 or render.ndim != 4:
            raise ValueError("query and render must have shape (B,C,H,W)")
        if query.shape[0] != render.shape[0] or query.shape[-2:] != render.shape[-2:]:
            raise ValueError("query and render must share batch and spatial dimensions")
        channels = min(int(query.shape[1]), int(render.shape[1]), self.channels)
        corr = dense_local_correlation(query[:, :channels], render[:, :channels], self.radius)
        raw = self.predict(torch.cat([corr, self._context(depth, corr)], dim=1))
        flow = torch.tanh(raw[:, :2]) * self.max_flow_px
        confidence = torch.sigmoid(raw[:, 2:3])
        return {"flow": flow.to(dtype=query.dtype), "confidence": confidence.to(dtype=query.dtype), "corr": corr}


def denseflow_supervision_loss(
    *,
    pred_flow: torch.Tensor,
    pred_confidence: torch.Tensor,
    gt_flow: torch.Tensor,
    valid: torch.Tensor,
    flow_weight: float = 1.0,
    confidence_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if pred_flow.shape != gt_flow.shape:
        raise ValueError(f"pred_flow and gt_flow shape mismatch: {pred_flow.shape} vs {gt_flow.shape}")
    if valid.ndim == 3:
        valid = valid[:, None]
    valid_f = valid.to(device=pred_flow.device, dtype=pred_flow.dtype)
    if valid_f.shape[-2:] != pred_flow.shape[-2:]:
        valid_f = F.interpolate(valid_f, size=pred_flow.shape[-2:], mode="nearest")
    conf = pred_confidence
    if conf.ndim == 3:
        conf = conf[:, None]
    if conf.shape[-2:] != pred_flow.shape[-2:]:
        conf = F.interpolate(conf.float(), size=pred_flow.shape[-2:], mode="bilinear", align_corners=False)
    epe_map = torch.linalg.vector_norm(pred_flow.float() - gt_flow.float(), dim=1, keepdim=True)
    denom = valid_f.sum().clamp(min=1.0)
    flow_loss = F.smooth_l1_loss(pred_flow.float() * valid_f, gt_flow.float() * valid_f, reduction="sum") / denom
    conf_target = (epe_map.detach() < 1.0).to(dtype=conf.dtype) * valid_f
    conf_loss = F.binary_cross_entropy(conf.clamp(1.0e-4, 1.0 - 1.0e-4), conf_target, reduction="none")
    conf_loss = (conf_loss * valid_f).sum() / denom
    loss = float(flow_weight) * flow_loss + float(confidence_weight) * conf_loss
    metrics = {
        "denseflow_loss": loss.detach(),
        "denseflow_flow_loss": flow_loss.detach(),
        "denseflow_conf_loss": conf_loss.detach(),
        "denseflow_flow_epe_px": ((epe_map * valid_f).sum() / denom).detach(),
        "denseflow_valid_frac": valid_f.mean().detach(),
        "denseflow_conf_mean": ((conf * valid_f).sum() / denom).detach(),
    }
    return loss, metrics
