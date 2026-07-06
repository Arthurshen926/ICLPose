from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.measurement_v1.stride4_fine_feature import local_correlation_logits


@dataclass(frozen=True)
class MeasurementPrediction:
    local_log_probs: torch.Tensor
    mean_xy_px: torch.Tensor
    cov_query_2x2: torch.Tensor
    mode_xy_px: torch.Tensor
    mode_probability: torch.Tensor
    entropy: torch.Tensor
    epe_px: torch.Tensor | None = None


def _xy_for_batch(sample_xy_px: torch.Tensor, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    xy = sample_xy_px.to(device=device, dtype=dtype)
    if xy.ndim == 2:
        xy = xy.unsqueeze(0).expand(int(batch_size), -1, -1)
    if xy.ndim != 3 or int(xy.shape[0]) != int(batch_size) or int(xy.shape[2]) != 2:
        raise ValueError("sample_xy_px must have shape (K,2) or (N,K,2)")
    return xy


def spatial_moments_from_logits(
    logits: torch.Tensor,
    sample_xy_px: torch.Tensor,
    *,
    covariance_floor_px2: float = 1e-4,
) -> MeasurementPrediction:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N,K)")
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    xy = _xy_for_batch(sample_xy_px, int(logits.shape[0]), device=logits.device, dtype=logits.dtype)
    mean = torch.sum(probs[..., None] * xy, dim=1)
    centered = xy - mean[:, None, :]
    cov = torch.einsum("nk,nki,nkj->nij", probs, centered, centered)
    eye = torch.eye(2, device=logits.device, dtype=logits.dtype).unsqueeze(0)
    cov = cov + eye * float(covariance_floor_px2)
    mode_idx = torch.argmax(probs, dim=1)
    batch = torch.arange(int(logits.shape[0]), device=logits.device)
    mode_xy = xy[batch, mode_idx]
    mode_probability = probs[batch, mode_idx]
    entropy = -torch.sum(probs * torch.clamp(log_probs, min=-1e12), dim=1)
    return MeasurementPrediction(
        local_log_probs=log_probs,
        mean_xy_px=mean,
        cov_query_2x2=cov,
        mode_xy_px=mode_xy,
        mode_probability=mode_probability,
        entropy=entropy,
    )


def _continuous_probability_at_target(
    logits: torch.Tensor,
    sample_xy_px: torch.Tensor,
    target_xy_px: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = spatial_moments_from_logits(logits, sample_xy_px)
    xy = _xy_for_batch(sample_xy_px, int(logits.shape[0]), device=logits.device, dtype=logits.dtype)
    target = target_xy_px.to(device=logits.device, dtype=logits.dtype).reshape(int(logits.shape[0]), 2)
    probs = torch.exp(pred.local_log_probs)
    out = []
    for row_idx in range(int(logits.shape[0])):
        row_xy = xy[row_idx]
        xs = torch.unique(row_xy[:, 0], sorted=True)
        ys = torch.unique(row_xy[:, 1], sorted=True)
        if int(xs.numel() * ys.numel()) != int(row_xy.shape[0]):
            distances = torch.linalg.norm(row_xy - target[row_idx].reshape(1, 2), dim=1)
            out.append(probs[row_idx, torch.argmin(distances)])
            continue
        grid = probs[row_idx].reshape(int(ys.numel()), int(xs.numel()))
        tx, ty = target[row_idx, 0], target[row_idx, 1]
        x1_idx = torch.searchsorted(xs, tx).clamp(1, int(xs.numel()) - 1)
        y1_idx = torch.searchsorted(ys, ty).clamp(1, int(ys.numel()) - 1)
        x0_idx = x1_idx - 1
        y0_idx = y1_idx - 1
        x0, x1 = xs[x0_idx], xs[x1_idx]
        y0, y1 = ys[y0_idx], ys[y1_idx]
        wx = ((tx - x0) / torch.clamp(x1 - x0, min=float(eps))).clamp(0.0, 1.0)
        wy = ((ty - y0) / torch.clamp(y1 - y0, min=float(eps))).clamp(0.0, 1.0)
        p00 = grid[y0_idx, x0_idx]
        p10 = grid[y0_idx, x1_idx]
        p01 = grid[y1_idx, x0_idx]
        p11 = grid[y1_idx, x1_idx]
        prob = p00 * (1.0 - wx) * (1.0 - wy) + p10 * wx * (1.0 - wy) + p01 * (1.0 - wx) * wy + p11 * wx * wy
        out.append(prob)
    return torch.stack(out).clamp_min(float(eps))


def continuous_window_nll_and_moments(
    logits: torch.Tensor,
    sample_xy_px: torch.Tensor,
    target_xy_px: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, MeasurementPrediction]:
    pred = spatial_moments_from_logits(logits, sample_xy_px)
    target = target_xy_px.to(device=logits.device, dtype=logits.dtype).reshape(int(logits.shape[0]), 2)
    prob = _continuous_probability_at_target(logits, sample_xy_px, target)
    per_row = -torch.log(prob)
    if confidence is not None:
        weights = confidence.detach().to(device=logits.device, dtype=logits.dtype).reshape(-1).clamp_min(0.0)
        loss = torch.sum(per_row * weights) / torch.sum(weights).clamp_min(1e-8)
    else:
        loss = torch.mean(per_row)
    epe = torch.linalg.norm(pred.mean_xy_px - target, dim=1)
    pred = MeasurementPrediction(
        local_log_probs=pred.local_log_probs,
        mean_xy_px=pred.mean_xy_px,
        cov_query_2x2=pred.cov_query_2x2,
        mode_xy_px=pred.mode_xy_px,
        mode_probability=pred.mode_probability,
        entropy=pred.entropy,
        epe_px=epe,
    )
    return loss, pred


class CorrelationMeasurementBranch(nn.Module):
    """Query-side measurement head from fixed render anchors and local correlation."""

    def __init__(self, *, search_radius_px: float = 8.0, step_px: float = 1.0, temperature: float = 1.0) -> None:
        super().__init__()
        self.search_radius_px = float(search_radius_px)
        self.step_px = float(step_px)
        self.temperature = float(temperature)

    def forward(
        self,
        *,
        query_features: torch.Tensor,
        render_features: torch.Tensor,
        query_centers_xy: torch.Tensor,
        render_anchor_xy: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> MeasurementPrediction:
        logits, xy = local_correlation_logits(
            query_features,
            render_features,
            query_centers_xy=query_centers_xy,
            render_anchor_xy=render_anchor_xy,
            image_width=int(image_width),
            image_height=int(image_height),
            search_radius_px=float(self.search_radius_px),
            step_px=float(self.step_px),
        )
        return spatial_moments_from_logits(logits / max(float(self.temperature), 1e-8), xy)


class SharedFeatureProjection(nn.Module):
    """Small shared projection for cached VFM/RGB features before correlation."""

    def __init__(self, *, input_dim: int, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        inp = int(input_dim)
        hidden = int(hidden_dim)
        out = int(output_dim)
        if inp <= 0 or hidden <= 0 or out <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(inp, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (B,C,H,W)")
        return F.normalize(self.net(features.float()), dim=1)


class Conv3FeatureProjection(nn.Module):
    """Small local patch encoder before correlation."""

    def __init__(self, *, input_dim: int, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        inp = int(input_dim)
        hidden = int(hidden_dim)
        out = int(output_dim)
        if inp <= 0 or hidden <= 0 or out <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(inp, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (B,C,H,W)")
        return F.normalize(self.net(features.float()), dim=1)
