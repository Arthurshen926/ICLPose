"""Projected selected-track map scoring utilities."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .selected_feature_map import SelectedTrackFeatureBank, score_query_with_selected_map_features


def _as_bk_pose(candidate_w2c: torch.Tensor) -> torch.Tensor:
    if candidate_w2c.ndim == 3:
        candidate_w2c = candidate_w2c[:, None]
    if candidate_w2c.ndim != 4 or candidate_w2c.shape[-2:] != (4, 4):
        raise ValueError("candidate_w2c must have shape (K,4,4) or (B,K,4,4)")
    return candidate_w2c.float()


def _intrinsics_for_batch(intrinsics: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    k = intrinsics.to(device=device).float()
    if k.ndim == 2:
        k = k.view(1, 3, 3).expand(batch_size, -1, -1)
    if k.ndim != 3 or k.shape[0] != batch_size or k.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape (3,3) or (B,3,3)")
    return k


def render_selected_track_feature_maps(
    bank: SelectedTrackFeatureBank,
    candidate_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    splat_radius: int = 0,
    z_eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project selected 3D track features into dense nearest-neighbor feature maps."""

    if bank.xyz is None:
        raise ValueError("SelectedTrackFeatureBank.xyz is required for projected rendering")
    poses = _as_bk_pose(candidate_w2c)
    device = poses.device
    features = bank.features.to(device=device).float()
    xyz = bank.xyz.to(device=device).float()
    if features.ndim != 2 or xyz.ndim != 2 or xyz.shape[0] != features.shape[0] or xyz.shape[1] != 3:
        raise ValueError("bank features must be (N,C) and xyz must be (N,3)")
    batch_size, num_candidates = poses.shape[:2]
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_hw must be positive")
    radius = max(0, int(splat_radius))
    k = _intrinsics_for_batch(intrinsics, batch_size, device)
    channels = features.shape[1]
    rendered = features.new_zeros((batch_size, num_candidates, channels, height, width))
    valid = torch.zeros((batch_size, num_candidates, 1, height, width), dtype=torch.bool, device=device)
    depth = features.new_full((batch_size, num_candidates, height, width), float("inf"))

    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=device)
    xyz_h = torch.cat([xyz, ones], dim=1).t()
    for bidx in range(batch_size):
        fx, fy = k[bidx, 0, 0], k[bidx, 1, 1]
        cx, cy = k[bidx, 0, 2], k[bidx, 1, 2]
        for cidx in range(num_candidates):
            cam = (poses[bidx, cidx] @ xyz_h).t()[:, :3]
            z = cam[:, 2]
            visible = z > float(z_eps)
            if not bool(visible.any()):
                continue
            u = torch.round(fx * cam[:, 0] / z.clamp_min(float(z_eps)) + cx).long()
            v = torch.round(fy * cam[:, 1] / z.clamp_min(float(z_eps)) + cy).long()
            inside = visible & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            for tidx in torch.nonzero(inside, as_tuple=False).flatten():
                center_y = int(v[tidx].detach().cpu())
                center_x = int(u[tidx].detach().cpu())
                zz = z[tidx]
                y0 = max(0, center_y - radius)
                y1 = min(height - 1, center_y + radius)
                x0 = max(0, center_x - radius)
                x1 = min(width - 1, center_x + radius)
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        if zz < depth[bidx, cidx, yy, xx]:
                            depth[bidx, cidx, yy, xx] = zz
                            rendered[bidx, cidx, :, yy, xx] = features[tidx]
                            valid[bidx, cidx, 0, yy, xx] = True
    return rendered, valid


def densify_projected_feature_maps(
    rendered: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Densify sparse projected feature maps by local valid-neighbor averaging."""

    radius = int(radius)
    if radius <= 0:
        return rendered, valid
    if rendered.ndim != 5 or valid.ndim != 5:
        raise ValueError("rendered and valid must have shape (B,K,C,H,W)/(B,K,1,H,W)")
    if rendered.shape[:2] != valid.shape[:2] or rendered.shape[-2:] != valid.shape[-2:] or valid.shape[2] != 1:
        raise ValueError("valid mask must match rendered batch/candidate/spatial dimensions and have one channel")
    bsz, num_candidates, channels, height, width = rendered.shape
    kernel = 2 * radius + 1
    valid_f = valid.to(device=rendered.device, dtype=rendered.dtype).reshape(bsz * num_candidates, 1, height, width)
    feat = rendered.reshape(bsz * num_candidates, channels, height, width) * valid_f
    scale = float(kernel * kernel)
    numerator = F.avg_pool2d(feat, kernel_size=kernel, stride=1, padding=radius) * scale
    denominator = F.avg_pool2d(valid_f, kernel_size=kernel, stride=1, padding=radius) * scale
    dense = numerator / denominator.clamp_min(1.0)
    dense_valid = denominator > 0.0
    return (
        dense.reshape(bsz, num_candidates, channels, height, width),
        dense_valid.reshape(bsz, num_candidates, 1, height, width),
    )


def score_projected_selected_track_bank(
    query_feature: torch.Tensor,
    bank: SelectedTrackFeatureBank,
    candidate_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    selector: torch.nn.Module,
    scorer: torch.nn.Module,
    map_adapter: torch.nn.Module | None = None,
    query_rgb: torch.Tensor | None = None,
    render_rgb: torch.Tensor | None = None,
    densify_radius: int = 0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Render/project a selected track bank and score it with the shared selector path."""

    rendered, render_valid = render_selected_track_feature_maps(
        bank,
        candidate_w2c,
        intrinsics,
        image_hw=image_hw,
    )
    rendered, render_valid = densify_projected_feature_maps(rendered, render_valid, radius=int(densify_radius))
    scores, aux = score_query_with_selected_map_features(
        query_feature,
        rendered.to(device=query_feature.device, dtype=query_feature.dtype),
        selector=selector,
        scorer=scorer,
        map_adapter=map_adapter,
        query_rgb=query_rgb,
        render_rgb=render_rgb,
        render_valid_mask=render_valid.to(device=query_feature.device),
    )
    aux = dict(aux)
    aux["render_valid_mask"] = render_valid.to(device=query_feature.device)
    aux["rendered_map_feature"] = rendered.to(device=query_feature.device, dtype=query_feature.dtype)
    return scores, aux


__all__ = ["densify_projected_feature_maps", "render_selected_track_feature_maps", "score_projected_selected_track_bank"]
