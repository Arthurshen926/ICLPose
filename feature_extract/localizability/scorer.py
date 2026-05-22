"""Pose-hypothesis scoring for POFD-FS."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_like(tensor: torch.Tensor | None, hw: tuple[int, int], *, mode: str) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.shape[-2:] == hw:
        return tensor
    return F.interpolate(tensor.float(), size=hw, mode=mode, align_corners=False if mode == "bilinear" else None)


def _shift_no_wrap(feature: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift feature maps without circular wraparound.

    Positive dx moves content right; positive dy moves content down.
    """
    if dy == 0 and dx == 0:
        return feature
    bsz, channels, height, width = feature.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(feature, (pad_left, pad_right, pad_top, pad_bottom))
    y0 = pad_bottom
    x0 = pad_right
    return padded[:, :, y0 : y0 + height, x0 : x0 + width]


def local_correlation_score_maps(
    query: torch.Tensor,
    render: torch.Tensor,
    *,
    radius: int = 4,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Return best local query-render similarity per candidate and pixel."""
    if query.ndim != 4 or render.ndim != 5:
        raise ValueError("query must be (B,C,H,W), render must be (B,K,C,H,W)")
    if query.shape[0] != render.shape[0] or query.shape[1] != render.shape[2]:
        raise ValueError("query/render batch and channel dimensions must match")
    if query.shape[-2:] != render.shape[-2:]:
        query = F.interpolate(query.float(), size=render.shape[-2:], mode="bilinear", align_corners=False)
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    temp = max(float(temperature), 1.0e-6)
    query_n = F.normalize(query.float(), dim=1, eps=1.0e-6)
    bsz, num_candidates, channels, height, width = render.shape
    render_flat = F.normalize(render.float().reshape(bsz * num_candidates, channels, height, width), dim=1, eps=1.0e-6)
    query_flat = query_n[:, None].expand(bsz, num_candidates, channels, height, width).reshape(
        bsz * num_candidates, channels, height, width
    )
    maps = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_no_wrap(render_flat, dy, dx)
            maps.append((query_flat * shifted).sum(dim=1))
    stacked = torch.stack(maps, dim=1)
    return (stacked / temp).amax(dim=1).reshape(bsz, num_candidates, height, width)


class PoseHypothesisScorer(nn.Module):
    """Score map/pose hypotheses from selected query and rendered features."""

    def __init__(
        self,
        *,
        mode: Literal["same_pixel", "local_corr", "pair_matcher_local"] = "local_corr",
        radius: int = 4,
        temperature: float = 0.05,
        pair_matcher: nn.Module | None = None,
        pair_matcher_stride: int = 8,
        pair_matcher_chunk_points: int = 65536,
        pair_matcher_candidate_chunk_size: int = 0,
        pair_matcher_offset_chunk_size: int = 0,
        pair_matcher_candidate_score_mode: str = "center_logprob_margin",
        pair_matcher_score_channel: int = 0,
    ):
        super().__init__()
        self.mode = str(mode)
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.pair_matcher = pair_matcher
        self.pair_matcher_stride = int(pair_matcher_stride)
        self.pair_matcher_chunk_points = int(pair_matcher_chunk_points)
        self.pair_matcher_candidate_chunk_size = int(pair_matcher_candidate_chunk_size)
        self.pair_matcher_offset_chunk_size = int(pair_matcher_offset_chunk_size)
        self.pair_matcher_candidate_score_mode = str(pair_matcher_candidate_score_mode)
        self.pair_matcher_score_channel = int(pair_matcher_score_channel)
        if self.mode not in {"same_pixel", "local_corr", "pair_matcher_local"}:
            raise ValueError(f"Unsupported scorer mode: {mode}")

    def forward(
        self,
        query: torch.Tensor,
        render: torch.Tensor,
        *,
        query_utility: torch.Tensor | None = None,
        render_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query.ndim != 4 or render.ndim != 5:
            raise ValueError("query must be (B,C,H,W), render must be (B,K,C,H,W)")
        if self.mode == "same_pixel":
            query_r = query
            if query_r.shape[-2:] != render.shape[-2:]:
                query_r = F.interpolate(query_r.float(), size=render.shape[-2:], mode="bilinear", align_corners=False)
            query_n = F.normalize(query_r.float(), dim=1, eps=1.0e-6)
            render_n = F.normalize(render.float(), dim=2, eps=1.0e-6)
            score_maps = (query_n[:, None] * render_n).sum(dim=2) / max(self.temperature, 1.0e-6)
        elif self.mode == "local_corr":
            score_maps = local_correlation_score_maps(
                query,
                render,
                radius=self.radius,
                temperature=self.temperature,
            )
        else:
            if self.pair_matcher is None:
                raise ValueError("mode='pair_matcher_local' requires pair_matcher")
            from feature_extract.tools.train_nvs_pose_feature_adapter import (  # noqa: PLC0415
                pair_matcher_local_candidate_score_maps,
            )

            score_maps_5d, valid_grid = pair_matcher_local_candidate_score_maps(
                self.pair_matcher,
                query,
                render,
                mask=render_valid_mask,
                radius=self.radius,
                stride=self.pair_matcher_stride,
                temperature=self.temperature,
                chunk_points=self.pair_matcher_chunk_points,
                candidate_chunk_size=self.pair_matcher_candidate_chunk_size,
                offset_chunk_size=self.pair_matcher_offset_chunk_size,
                candidate_score_mode=self.pair_matcher_candidate_score_mode,
            )
            bsz, num_candidates, channels, height, width = score_maps_5d.shape
            score_channel = int(self.pair_matcher_score_channel)
            if score_channel < 0:
                score_channel = channels + score_channel
            if score_channel < 0 or score_channel >= channels:
                raise ValueError(f"pair_matcher_score_channel={self.pair_matcher_score_channel} outside [0,{channels})")
            weight = _resize_like(query_utility, (height, width), mode="bilinear")
            if weight is None:
                weight = score_maps_5d.new_ones((bsz, 1, height, width))
            else:
                weight = weight.to(device=score_maps_5d.device, dtype=score_maps_5d.dtype).clamp_min(0.0)
            valid_weight = valid_grid.to(device=score_maps_5d.device, dtype=score_maps_5d.dtype).clamp(0.0, 1.0)
            combined_weight = weight[:, None, 0] * valid_weight
            raw_weight_sum = combined_weight.flatten(2).sum(dim=-1)
            denom = raw_weight_sum.clamp_min(1.0)
            scores = (score_maps_5d[:, :, score_channel] * combined_weight).flatten(2).sum(dim=-1) / denom
            return scores, {
                "score_maps": score_maps_5d,
                "valid_mask": raw_weight_sum > 0.0,
                "weight_sum": raw_weight_sum,
            }

        bsz, num_candidates, height, width = score_maps.shape
        weight = _resize_like(query_utility, (height, width), mode="bilinear")
        if weight is None:
            weight = score_maps.new_ones((bsz, 1, height, width))
        else:
            weight = weight.to(device=score_maps.device, dtype=score_maps.dtype).clamp_min(0.0)
        valid = _resize_like(render_valid_mask, (height, width), mode="nearest")
        if valid is None:
            valid_weight = score_maps.new_ones((bsz, num_candidates, height, width))
        else:
            if valid.ndim == 5:
                valid = valid[:, :, 0]
            if valid.ndim == 4 and valid.shape[1] == 1:
                valid = valid[:, None, 0].expand(-1, num_candidates, -1, -1)
            valid_weight = valid.to(device=score_maps.device, dtype=score_maps.dtype).clamp(0.0, 1.0)
        combined_weight = weight[:, None, 0] * valid_weight
        raw_weight_sum = combined_weight.flatten(2).sum(dim=-1)
        denom = raw_weight_sum.clamp_min(1.0)
        scores = (score_maps * combined_weight).flatten(2).sum(dim=-1) / denom
        valid_mask = raw_weight_sum > 0.0
        return scores, {
            "score_maps": score_maps,
            "valid_mask": valid_mask,
            "weight_sum": raw_weight_sum,
        }
