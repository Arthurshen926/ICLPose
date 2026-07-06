from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def _bin_xy(device: torch.device, dtype: torch.dtype, bins: int = 8) -> torch.Tensor:
    values = torch.arange(int(bins * bins), device=device, dtype=dtype)
    return torch.stack([torch.remainder(values, bins) + 0.5, torch.floor(values / bins) + 0.5], dim=1)


def continuous_fine_nll(
    logits: torch.Tensor,
    target_xy_bins: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Continuous bilinear likelihood over 8x8 spatial fine logits.

    target_xy_bins is in bin-coordinate units with bin centers at k+0.5.
    """

    if logits.ndim != 2 or int(logits.shape[1]) < 64:
        raise ValueError("logits must have shape (N, >=64)")
    target = target_xy_bins.to(device=logits.device, dtype=logits.dtype).reshape(-1, 2)
    if int(target.shape[0]) != int(logits.shape[0]):
        raise ValueError("target_xy_bins must contain one coordinate per logit row")
    spatial_logits = logits[:, :64]
    log_probs = F.log_softmax(spatial_logits, dim=1).reshape(-1, 8, 8)
    x = target[:, 0] - 0.5
    y = target[:, 1] - 0.5
    x0 = torch.floor(x).long().clamp(0, 7)
    y0 = torch.floor(y).long().clamp(0, 7)
    x1 = (x0 + 1).clamp(0, 7)
    y1 = (y0 + 1).clamp(0, 7)
    wx = (x - x0.to(dtype=logits.dtype)).clamp(0.0, 1.0)
    wy = (y - y0.to(dtype=logits.dtype)).clamp(0.0, 1.0)
    rows = torch.arange(int(logits.shape[0]), device=logits.device)
    p00 = torch.exp(log_probs[rows, y0, x0])
    p10 = torch.exp(log_probs[rows, y0, x1])
    p01 = torch.exp(log_probs[rows, y1, x0])
    p11 = torch.exp(log_probs[rows, y1, x1])
    prob = p00 * (1.0 - wx) * (1.0 - wy) + p10 * wx * (1.0 - wy) + p01 * (1.0 - wx) * wy + p11 * wx * wy
    per_row = -torch.log(prob.clamp_min(float(eps)))
    if confidence is not None:
        weights = confidence.detach().to(device=logits.device, dtype=logits.dtype).reshape(-1).clamp_min(0.0)
        if int(weights.shape[0]) != int(logits.shape[0]):
            raise ValueError("confidence must contain one value per logit row")
        denom = torch.sum(weights).clamp_min(float(eps))
        loss = torch.sum(per_row * weights) / denom
    else:
        loss = torch.mean(per_row)
    with torch.no_grad():
        pred = local_map_refined_xy(logits)
        epe = torch.linalg.norm(pred - target, dim=1)
    return loss, {
        "valid_count": float(int(logits.shape[0])),
        "nll": float(loss.detach().cpu().item()),
        "continuous_epe_bins": float(torch.mean(epe).detach().cpu().item()),
    }


def local_map_refined_xy(logits: torch.Tensor, *, radius: int = 1) -> torch.Tensor:
    """Return MAP-neighborhood soft coordinate, not global softargmax."""

    if logits.ndim != 2 or int(logits.shape[1]) < 64:
        raise ValueError("logits must have shape (N, >=64)")
    spatial = logits[:, :64]
    probs = F.softmax(spatial, dim=1)
    coords = _bin_xy(logits.device, logits.dtype)
    mode = torch.argmax(probs, dim=1)
    mode_xy = coords[mode]
    dx = torch.abs(coords[None, :, 0] - mode_xy[:, None, 0])
    dy = torch.abs(coords[None, :, 1] - mode_xy[:, None, 1])
    mask = (dx <= float(radius)) & (dy <= float(radius))
    local_probs = torch.where(mask, probs, torch.zeros_like(probs))
    denom = torch.sum(local_probs, dim=1, keepdim=True).clamp_min(1e-8)
    local_probs = local_probs / denom
    return local_probs @ coords


def continuous_fine_loss_and_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, Mapping[str, float]]:
    target = labels.long().reshape(-1)
    valid = (target >= 0) & (target < 64)
    if int(torch.count_nonzero(valid).detach().cpu().item()) == 0:
        return None, {"valid_count": 0.0}
    coords = _bin_xy(logits.device, logits.dtype)
    weights = None if confidence is None else confidence[valid]
    return continuous_fine_nll(logits[valid], coords[target[valid]], confidence=weights)
