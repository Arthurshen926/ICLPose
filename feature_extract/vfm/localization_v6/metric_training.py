"""Differentiable V6 local-correlation supervision."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def sample_feature_grid(
    feature_map: torch.Tensor, xy: torch.Tensor
) -> torch.Tensor:
    """Bilinearly sample a BCHW tensor at per-batch pixel-center coordinates."""

    if feature_map.ndim != 4 or xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("expected BCHW features and BxNx2 coordinates")
    if feature_map.shape[0] != xy.shape[0]:
        raise ValueError("batch dimensions differ")
    height, width = feature_map.shape[-2:]
    normalized = torch.stack(
        [
            (xy[..., 0] + 0.5) * 2.0 / width - 1.0,
            (xy[..., 1] + 0.5) * 2.0 / height - 1.0,
        ],
        dim=-1,
    )
    sampled = F.grid_sample(
        feature_map,
        normalized[:, :, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, :, :, 0].transpose(1, 2)


def local_correlation_training_loss(
    map_output: Mapping[str, torch.Tensor],
    query_output: Mapping[str, torch.Tensor],
    *,
    map_xy: torch.Tensor,
    proposal_xy: torch.Tensor,
    target_xy: torch.Tensor,
    positive_mask: torch.Tensor,
    query_matchable_mask: torch.Tensor | None = None,
    map_descriptor_override: torch.Tensor | None = None,
    feature_key: str = "fine",
    radius: int = 4,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Train full displacement posterior, explicit null and uncertainty.

    Positives use the nearest discrete displacement cell.  Wrong maplets,
    occluded and out-of-window samples use the explicit null class.
    """

    if feature_key not in {"fine", "middle", "coarse"}:
        raise ValueError("feature_key must be fine, middle or coarse")
    map_features = map_output[feature_key]
    query_features = query_output[feature_key]
    if map_xy.shape != proposal_xy.shape or map_xy.shape != target_xy.shape:
        raise ValueError("coordinate tensors must share shape BxNx2")
    batch, count = map_xy.shape[:2]
    device = map_xy.device
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing="ij",
    )
    offsets = torch.stack(
        [offsets_x.reshape(-1), offsets_y.reshape(-1)], dim=1
    ).to(map_xy.dtype)
    map_descriptors = F.normalize(
        sample_feature_grid(map_features, map_xy)
        if map_descriptor_override is None
        else map_descriptor_override,
        dim=-1,
    )
    if map_descriptors.shape[:2] != map_xy.shape[:2]:
        raise ValueError("map descriptor override differs from coordinates")
    patch_xy = proposal_xy[:, :, None] + offsets[None, None]
    patch_descriptors = sample_feature_grid(
        query_features,
        patch_xy.reshape(batch, -1, 2),
    ).reshape(batch, count, offsets.shape[0], -1)
    patch_descriptors = F.normalize(patch_descriptors, dim=-1)
    correlation_logits = torch.sum(
        map_descriptors[:, :, None] * patch_descriptors, dim=-1
    ) / float(temperature)
    null_map = F.interpolate(
        query_output["null_logits"],
        size=query_features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    null_logits = sample_feature_grid(null_map, proposal_xy)[..., 0]
    logits = torch.cat([correlation_logits, null_logits[:, :, None]], dim=-1)
    true_displacement = target_xy - proposal_xy
    nearest = torch.argmin(
        torch.sum(
            (offsets[None, None] - true_displacement[:, :, None]) ** 2,
            dim=-1,
        ),
        dim=-1,
    )
    inside = (
        (torch.abs(true_displacement[..., 0]) <= radius + 0.5)
        & (torch.abs(true_displacement[..., 1]) <= radius + 0.5)
    )
    is_positive = positive_mask.bool() & inside
    null_class = torch.full_like(nearest, offsets.shape[0])
    target_class = torch.where(is_positive, nearest, null_class)
    correlation_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target_class.reshape(-1)
    )
    matchability_map = F.interpolate(
        query_output["matchability_logits"],
        size=query_features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    matchability_logits = sample_feature_grid(
        matchability_map, proposal_xy
    )[..., 0]
    matchable_target = (
        is_positive
        if query_matchable_mask is None
        else query_matchable_mask.bool()
    )
    matchability_loss = F.binary_cross_entropy_with_logits(
        matchability_logits, matchable_target.float()
    )
    probabilities = torch.softmax(logits, dim=-1)
    displacement_probability = probabilities[..., :-1]
    probability_mass = displacement_probability.sum(dim=-1).clamp_min(1e-8)
    mean = torch.sum(
        displacement_probability[..., None] * offsets[None, None], dim=-2
    ) / probability_mass[..., None]
    centered = offsets[None, None] - mean[:, :, None]
    covariance = torch.einsum(
        "bnk,bnki,bnkj->bnij",
        displacement_probability / probability_mass[..., None],
        centered,
        centered,
    )
    covariance = covariance + torch.eye(2, device=device)[None, None] * 1e-3
    error = mean - true_displacement
    inverse = torch.linalg.inv(covariance)
    mahalanobis = torch.einsum("bni,bnij,bnj->bn", error, inverse, error)
    logdet = torch.logdet(covariance)
    if torch.any(is_positive):
        displacement_nll = 0.5 * (
            mahalanobis[is_positive] + logdet[is_positive]
        ).mean()
        flow_epe = torch.linalg.norm(error[is_positive], dim=-1).mean()
    else:
        displacement_nll = logits.sum() * 0.0
        flow_epe = logits.sum() * 0.0
    total = correlation_nll + 0.25 * matchability_loss + 0.1 * displacement_nll
    return {
        "total": total,
        "correlation_nll": correlation_nll,
        "matchability": matchability_loss,
        "displacement_nll": displacement_nll,
        "flow_epe": flow_epe,
        "null_probability": probabilities[..., -1].mean(),
        "positive_fraction": is_positive.float().mean(),
        "mean_displacement": mean,
        "displacement_covariance": covariance,
        "positive_mask": is_positive,
    }


def analytic_one_step_pose_loss(
    mean_displacement: torch.Tensor,
    covariance: torch.Tensor,
    projection_jacobian: torch.Tensor,
    valid_mask: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    displacement_scale: float,
    damping: float = 1e-3,
) -> torch.Tensor:
    """Differentiate through the same weighted joint-6DoF normal equations."""

    if (
        mean_displacement.ndim != 3
        or covariance.shape != (*mean_displacement.shape[:2], 2, 2)
        or projection_jacobian.shape
        != (*mean_displacement.shape[:2], 2, 6)
    ):
        raise ValueError("invalid analytic pose-loss tensor shapes")
    batch = mean_displacement.shape[0]
    estimates = []
    identity2 = torch.eye(
        2, dtype=mean_displacement.dtype, device=mean_displacement.device
    )
    identity6 = torch.eye(
        6, dtype=mean_displacement.dtype, device=mean_displacement.device
    )
    for row in range(batch):
        keep = valid_mask[row].bool()
        if int(torch.sum(keep)) < 6:
            estimates.append(target_delta[row] * 0.0)
            continue
        displacement = mean_displacement[row, keep] * float(displacement_scale)
        cov = covariance[row, keep] * float(displacement_scale) ** 2
        information = torch.linalg.inv(cov + identity2[None] * 0.25)
        jacobian = projection_jacobian[row, keep]
        # Form J^T W J explicitly while preserving autograd and joint
        # cross-axis terms.
        normal = torch.sum(
            jacobian.transpose(1, 2) @ information @ jacobian, dim=0
        )
        rhs = torch.sum(
            (
                jacobian.transpose(1, 2)
                @ information
                @ displacement[:, :, None]
            )[:, :, 0],
            dim=0,
        )
        estimates.append(
            torch.linalg.solve(normal + identity6 * float(damping), rhs)
        )
    estimate = torch.stack(estimates)
    return F.smooth_l1_loss(estimate, target_delta)
