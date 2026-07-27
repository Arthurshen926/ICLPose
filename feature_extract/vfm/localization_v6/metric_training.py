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
    map_descriptor_modes_override: torch.Tensor | None = None,
    map_mode_log_prior: torch.Tensor | None = None,
    map_uncertainty_override: torch.Tensor | None = None,
    feature_key: str = "fine",
    radius: int = 4,
    temperature: float = 0.07,
    pair_null_logit: float = 0.0,
    uncertainty_temperature_scale: float = 2.0,
    uncertainty_null_scale: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Train full displacement posterior, explicit null and uncertainty.

    Positives use the nearest discrete displacement cell.  Wrong maplets,
    occluded and out-of-window samples use the explicit null class.
    """

    if feature_key not in {"fine", "middle", "coarse"}:
        raise ValueError("feature_key must be fine, middle or coarse")
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
    if (
        map_descriptor_override is not None
        and map_descriptor_modes_override is not None
    ):
        raise ValueError("single and multi-mode descriptor overrides are exclusive")
    map_descriptors = None
    if map_descriptor_modes_override is None:
        map_descriptors = F.normalize(
            sample_feature_grid(map_output[feature_key], map_xy)
            if map_descriptor_override is None
            else map_descriptor_override,
            dim=-1,
        )
        if map_descriptors.shape[:2] != map_xy.shape[:2]:
            raise ValueError(
                "map descriptor override differs from coordinates"
            )
    patch_xy = proposal_xy[:, :, None] + offsets[None, None]
    patch_descriptors = sample_feature_grid(
        query_features,
        patch_xy.reshape(batch, -1, 2),
    ).reshape(batch, count, offsets.shape[0], -1)
    patch_descriptors = F.normalize(patch_descriptors, dim=-1)
    if map_uncertainty_override is None:
        map_uncertainty = torch.zeros(
            (batch, count),
            dtype=patch_descriptors.dtype,
            device=device,
        )
    else:
        map_uncertainty = map_uncertainty_override.to(
            dtype=patch_descriptors.dtype, device=device
        )
        if map_uncertainty.shape != (batch, count):
            raise ValueError("map uncertainty override differs from coordinates")
        map_uncertainty = map_uncertainty.clamp(0.0, 1.0)
    effective_temperature = max(float(temperature), 1e-4) * (
        1.0
        + float(uncertainty_temperature_scale) * map_uncertainty
    )
    if map_descriptor_modes_override is None:
        assert map_descriptors is not None
        correlation_logits = torch.sum(
            map_descriptors[:, :, None] * patch_descriptors, dim=-1
        ) / effective_temperature[:, :, None]
    else:
        mode_descriptors = F.normalize(map_descriptor_modes_override, dim=-1)
        if mode_descriptors.shape[:2] != map_xy.shape[:2]:
            raise ValueError("mode descriptor override differs from coordinates")
        if map_mode_log_prior is None or map_mode_log_prior.shape != mode_descriptors.shape[:3]:
            raise ValueError("mode log prior differs from mode descriptors")
        mode_logits = torch.einsum(
            "bnkc,bnoc->bnko", mode_descriptors, patch_descriptors
        ) / effective_temperature[:, :, None, None]
        correlation_logits = torch.logsumexp(
            mode_logits + map_mode_log_prior[:, :, :, None], dim=2
        )
    candidate_valid = (
        (patch_xy[..., 0] >= 0.0)
        & (patch_xy[..., 0] <= query_features.shape[-1] - 1)
        & (patch_xy[..., 1] >= 0.0)
        & (patch_xy[..., 1] <= query_features.shape[-2] - 1)
    )
    valid_count = candidate_valid.sum(dim=-1).clamp_min(1).to(
        correlation_logits.dtype
    )
    negative_log_partition = 0.5 / (
        query_features.shape[1]
        * effective_temperature
        * effective_temperature
    )
    correlation_logits = (
        correlation_logits
        - torch.log(valid_count)[..., None]
        - negative_log_partition[..., None]
    )
    correlation_logits = torch.where(
        candidate_valid,
        correlation_logits,
        torch.full_like(correlation_logits, -1e4),
    )
    matchability_map = F.interpolate(
        query_output["matchability_logits"],
        size=query_features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    candidate_matchability_logits = sample_feature_grid(
        matchability_map,
        patch_xy.reshape(batch, -1, 2),
    ).reshape(batch, count, offsets.shape[0])
    # Candidate-wise matchability changes the relative offset probabilities.
    # Adding the centre value to every offset would cancel in the softmax.
    correlation_logits = correlation_logits + F.logsigmoid(
        candidate_matchability_logits
    )
    null_logits = torch.full(
        (batch, count),
        float(pair_null_logit),
        dtype=correlation_logits.dtype,
        device=device,
    )
    null_logits = null_logits + (
        float(uncertainty_null_scale) * map_uncertainty
    )
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
    per_sample_correlation_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_class.reshape(-1),
        reduction="none",
    ).reshape(batch, count)
    positive_correlation_nll = (
        per_sample_correlation_nll[is_positive].mean()
        if torch.any(is_positive)
        else logits.sum() * 0.0
    )
    is_null = ~is_positive
    null_correlation_nll = (
        per_sample_correlation_nll[is_null].mean()
        if torch.any(is_null)
        else logits.sum() * 0.0
    )
    if torch.any(is_positive) and torch.any(is_null):
        # Episode composition changes with occlusion and search level.  Give
        # spatial positives and null outcomes equal objective mass so that a
        # lower total loss cannot be obtained by sacrificing correct modes.
        correlation_nll = 0.5 * (
            positive_correlation_nll + null_correlation_nll
        )
    else:
        correlation_nll = per_sample_correlation_nll.mean()
    matchable_target = (
        is_positive
        if query_matchable_mask is None
        else query_matchable_mask.bool()
    )
    supervision_xy = torch.where(
        matchable_target[..., None], target_xy, proposal_xy
    )
    matchability_logits = sample_feature_grid(
        matchability_map, supervision_xy
    )[..., 0]
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
        target_direction = true_displacement[is_positive]
        predicted_direction = mean[is_positive]
        nonzero = torch.linalg.norm(target_direction, dim=-1) > 1e-4
        if torch.any(nonzero):
            direction_cosine = F.cosine_similarity(
                predicted_direction[nonzero],
                target_direction[nonzero],
                dim=-1,
            ).mean()
        else:
            direction_cosine = logits.sum() * 0.0
        mode_recall = (
            torch.argmax(displacement_probability[is_positive], dim=-1)
            == nearest[is_positive]
        ).float().mean()
        predicted_mode = offsets[
            torch.argmax(
                displacement_probability[is_positive], dim=-1
            )
        ]
        target_mode = offsets[nearest[is_positive]]
        mode_recall_radius1 = (
            torch.max(
                torch.abs(predicted_mode - target_mode), dim=-1
            ).values
            <= 1.0
        ).float().mean()
    else:
        displacement_nll = logits.sum() * 0.0
        flow_epe = logits.sum() * 0.0
        direction_cosine = logits.sum() * 0.0
        mode_recall = logits.sum() * 0.0
        mode_recall_radius1 = logits.sum() * 0.0
    total = correlation_nll + 0.25 * matchability_loss + 0.1 * displacement_nll
    return {
        "total": total,
        "correlation_nll": correlation_nll,
        "positive_correlation_nll": positive_correlation_nll,
        "null_correlation_nll": null_correlation_nll,
        "matchability": matchability_loss,
        "displacement_nll": displacement_nll,
        "flow_epe": flow_epe,
        "direction_cosine": direction_cosine,
        "mode_recall": mode_recall,
        "mode_recall_radius1": mode_recall_radius1,
        "in_window_fraction": inside.float().mean(),
        "null_probability": probabilities[..., -1].mean(),
        "null_probability_per_sample": probabilities[..., -1],
        "null_target": ~is_positive,
        "inside_mask": inside,
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
