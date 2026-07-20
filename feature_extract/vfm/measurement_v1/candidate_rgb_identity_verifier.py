"""Independent real-RGB evidence for top-L landmark identity hypotheses.

The verifier deliberately does not consume retrieval scores, candidate ranks, pose
statistics, or geometry classifier posteriors.  It returns candidate-specific log
likelihood ratios.  Missing RGB observations are represented by a zero log
likelihood ratio, so they cannot move probability mass by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
    cost_volume_quality_features,
    template_search_cost_volume_logits,
)


SUPPORT_VIEW_MIXTURE_CONTRACT = (
    "frozen_posterior_with_neutral_missing_mass_v1"
)
MEASUREMENT_MODE_SUCCESS_SEMANTICS = (
    "P(predicted_spatial_mode_gt_pose_projection_residual_le_2px)"
)
GT_POSE_SPATIAL_DENSITY_SEMANTICS = (
    "normalized_k_plus_dustbin_gt_pose_projected_offset_v1"
)
POSE_VIEW_MIXTURE_SEMANTICS = (
    "candidate_masked_softmax_rgb_pair_hidden_gt_pose_density_nll_v1"
)
IDENTITY_CONTEXT_SEMANTICS = (
    "per_view_aligned_broad_real_rgb_context_identity_residual_no_spatial_density_v1"
)
IDENTITY_CONTEXT_LAYOUT_SEMANTICS = (
    "per_view_anchor_aligned_broad_real_rgb_relative_layout_separate_encoder_identity_residual_no_spatial_density_v1"
)


def normalized_spatial_log_probabilities_with_dustbin(
    spatial_logits: torch.Tensor,
    non_dustbin_logits: torch.Tensor,
) -> torch.Tensor:
    """Return a normalized categorical density over K offsets and dustbin.

    The non-dustbin head controls only whether the finite local support is
    usable. Conditional offset probabilities remain a separate categorical
    distribution. This makes the exported measurement a single probability
    object instead of two independently interpreted classifier outputs.
    """

    if spatial_logits.ndim != 2:
        raise ValueError("spatial_logits must have shape (N,K)")
    non_dustbin = non_dustbin_logits.to(
        device=spatial_logits.device, dtype=spatial_logits.dtype
    ).reshape(-1)
    if int(non_dustbin.numel()) != int(spatial_logits.shape[0]):
        raise ValueError(
            "non_dustbin_logits must contain one value per spatial map"
        )
    local = (
        F.logsigmoid(non_dustbin)[:, None]
        + F.log_softmax(spatial_logits, dim=1)
    )
    dustbin = F.logsigmoid(-non_dustbin)[:, None]
    return torch.cat([local, dustbin], dim=1)


@dataclass(frozen=True)
class RGBCandidateIdentityPrediction:
    view_identity_logits: torch.Tensor
    view_log_likelihood_ratios: torch.Tensor
    view_measurement_validity_logits: torch.Tensor
    view_measurement_validity_probabilities: torch.Tensor
    view_spatial_logits: torch.Tensor
    view_spatial_offsets_xy: torch.Tensor
    view_pose_mixture_logits: torch.Tensor
    view_pose_mixture_probabilities: torch.Tensor
    candidate_pose_view_probabilities: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    candidate_conditional_probabilities: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_measured: torch.Tensor
    candidate_view_coverage: torch.Tensor
    measured_set_availability_logit: torch.Tensor
    measured_set_availability_probability: torch.Tensor


def normalize_pose_view_mixture_logits(
    view_logits: torch.Tensor,
    *,
    pair_group_indices: torch.Tensor,
    pair_candidate_indices: torch.Tensor,
    pair_view_slots: torch.Tensor,
    batch_size: int,
    candidate_count: int,
    max_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize pose-view weights independently inside every candidate.

    The operation is permutation equivariant over input rows and support-view
    slots. Candidates without an RGB observation retain zero probability mass.
    """

    if int(batch_size) <= 0 or int(candidate_count) <= 0 or int(max_views) <= 0:
        raise ValueError("pose-view mixture dimensions must be positive")
    logits = view_logits.reshape(-1).float()
    groups = pair_group_indices.to(device=logits.device, dtype=torch.long).reshape(-1)
    candidates = pair_candidate_indices.to(
        device=logits.device, dtype=torch.long
    ).reshape(-1)
    slots = pair_view_slots.to(device=logits.device, dtype=torch.long).reshape(-1)
    if not (
        int(logits.numel())
        == int(groups.numel())
        == int(candidates.numel())
        == int(slots.numel())
    ):
        raise ValueError("pose-view logits and pair indices must share row count")
    if bool(torch.any(groups < 0)) or bool(torch.any(groups >= int(batch_size))):
        raise ValueError("pair_group_indices contains an out-of-range value")
    if bool(torch.any(candidates < 0)) or bool(
        torch.any(candidates >= int(candidate_count))
    ):
        raise ValueError("pair_candidate_indices contains an out-of-range value")
    if bool(torch.any(slots < 0)) or bool(torch.any(slots >= int(max_views))):
        raise ValueError("pair_view_slots contains an out-of-range value")

    linear = (groups * int(candidate_count) + candidates) * int(max_views) + slots
    if int(torch.unique(linear).numel()) != int(linear.numel()):
        raise ValueError("a group/candidate/view slot may be populated at most once")
    total_slots = int(batch_size) * int(candidate_count) * int(max_views)
    padded_logits = torch.index_copy(
        torch.full(
            (total_slots,), -1e9, device=logits.device, dtype=torch.float32
        ),
        0,
        linear,
        logits,
    ).reshape(int(batch_size), int(candidate_count), int(max_views))
    present_flat = torch.zeros(
        (total_slots,), device=logits.device, dtype=torch.bool
    )
    present_flat[linear] = True
    present = present_flat.reshape(
        int(batch_size), int(candidate_count), int(max_views)
    )
    padded_probabilities = F.softmax(padded_logits, dim=2)
    any_present = torch.any(present, dim=2, keepdim=True)
    padded_probabilities = torch.where(
        any_present,
        padded_probabilities * present.to(dtype=torch.float32),
        torch.zeros_like(padded_probabilities),
    )
    pair_probabilities = padded_probabilities.reshape(-1)[linear]
    return pair_probabilities, padded_probabilities


def aggregate_pose_view_spatial_log_probabilities(
    view_joint_log_probabilities: torch.Tensor,
    view_probabilities: torch.Tensor,
    *,
    pair_group_indices: torch.Tensor,
    pair_candidate_indices: torch.Tensor,
    pair_view_slots: torch.Tensor,
    batch_size: int,
    candidate_count: int,
    max_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize normalized K+1 view densities inside each candidate."""

    joint = view_joint_log_probabilities.float()
    if joint.ndim != 2:
        raise ValueError("view joint log probabilities must have shape (N,K+1)")
    probabilities = view_probabilities.to(
        device=joint.device, dtype=torch.float32
    ).reshape(-1)
    groups = pair_group_indices.to(device=joint.device, dtype=torch.long).reshape(-1)
    candidates = pair_candidate_indices.to(
        device=joint.device, dtype=torch.long
    ).reshape(-1)
    slots = pair_view_slots.to(device=joint.device, dtype=torch.long).reshape(-1)
    if not (
        int(joint.shape[0])
        == int(probabilities.numel())
        == int(groups.numel())
        == int(candidates.numel())
        == int(slots.numel())
    ):
        raise ValueError("pose-view spatial rows and pair indices are misaligned")
    if bool(torch.any(~torch.isfinite(probabilities))) or bool(
        torch.any(probabilities < 0.0)
    ):
        raise ValueError("pose-view mixture probabilities are invalid")
    if bool(torch.any(groups < 0)) or bool(torch.any(groups >= int(batch_size))):
        raise ValueError("pair_group_indices contains an out-of-range value")
    if bool(torch.any(candidates < 0)) or bool(
        torch.any(candidates >= int(candidate_count))
    ):
        raise ValueError("pair_candidate_indices contains an out-of-range value")
    if bool(torch.any(slots < 0)) or bool(torch.any(slots >= int(max_views))):
        raise ValueError("pair_view_slots contains an out-of-range value")

    linear = (groups * int(candidate_count) + candidates) * int(max_views) + slots
    if int(torch.unique(linear).numel()) != int(linear.numel()):
        raise ValueError("a group/candidate/view slot may be populated at most once")
    total_slots = int(batch_size) * int(candidate_count) * int(max_views)
    padded_joint = torch.index_copy(
        torch.full(
            (total_slots, int(joint.shape[1])),
            -torch.inf,
            device=joint.device,
            dtype=torch.float32,
        ),
        0,
        linear,
        joint,
    ).reshape(
        int(batch_size),
        int(candidate_count),
        int(max_views),
        int(joint.shape[1]),
    )
    padded_probability = torch.index_copy(
        torch.zeros((total_slots,), device=joint.device, dtype=torch.float32),
        0,
        linear,
        probabilities,
    ).reshape(int(batch_size), int(candidate_count), int(max_views))
    present_flat = torch.zeros(
        (total_slots,), device=joint.device, dtype=torch.bool
    )
    present_flat[linear] = True
    present = present_flat.reshape(
        int(batch_size), int(candidate_count), int(max_views)
    )
    measured = torch.any(present, dim=2)
    mass = torch.sum(padded_probability, dim=2)
    if bool(torch.any(torch.abs(mass[measured] - 1.0) > 2e-5)):
        raise ValueError("measured candidate pose-view probability mass must equal one")
    terms = torch.where(
        present[..., None] & (padded_probability[..., None] > 0.0),
        torch.log(padded_probability.clamp_min(1e-30))[..., None] + padded_joint,
        torch.full_like(padded_joint, -torch.inf),
    )
    mixture = torch.logsumexp(terms, dim=2)
    mixture = torch.where(
        measured[..., None], mixture, torch.zeros_like(mixture)
    )
    return mixture, measured


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != valid.shape:
        raise ValueError("candidate logits and valid mask must share shape")
    masked = torch.where(valid, logits, torch.full_like(logits, -1e9))
    probabilities = F.softmax(masked, dim=1)
    any_valid = torch.any(valid, dim=1, keepdim=True)
    return torch.where(any_valid, probabilities * valid.to(logits.dtype), torch.zeros_like(probabilities))


def aggregate_view_log_likelihood_ratios(
    view_log_likelihood_ratios: torch.Tensor,
    view_hidden: torch.Tensor,
    *,
    pair_view_probabilities: torch.Tensor,
    pair_group_indices: torch.Tensor,
    pair_candidate_indices: torch.Tensor,
    pair_view_slots: torch.Tensor,
    batch_size: int,
    candidate_count: int,
    max_views: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Marginalize a fixed-size view mixture with neutral missing-view mass.

    A missing view contributes likelihood ratio one, or log-likelihood ratio zero.
    This prevents an unavailable image from being renormalized into evidence for a
    different candidate.
    """

    if int(max_views) <= 0 or int(candidate_count) <= 0 or int(batch_size) <= 0:
        raise ValueError("batch_size, candidate_count, and max_views must be positive")
    view_logits = view_log_likelihood_ratios.reshape(-1).float()
    view_prior = pair_view_probabilities.to(
        device=view_logits.device, dtype=torch.float32
    ).reshape(-1)
    hidden = view_hidden.float()
    if hidden.ndim != 2 or int(hidden.shape[0]) != int(view_logits.numel()):
        raise ValueError("view_hidden must have one row per view log-likelihood ratio")
    groups = pair_group_indices.to(device=view_logits.device, dtype=torch.long).reshape(-1)
    candidates = pair_candidate_indices.to(device=view_logits.device, dtype=torch.long).reshape(-1)
    slots = pair_view_slots.to(device=view_logits.device, dtype=torch.long).reshape(-1)
    if not (
        int(groups.numel())
        == int(candidates.numel())
        == int(slots.numel())
        == int(view_logits.numel())
        == int(view_prior.numel())
    ):
        raise ValueError("pair indices must have one entry per view")
    if bool(torch.any(~torch.isfinite(view_prior))) or bool(torch.any(view_prior < 0.0)):
        raise ValueError("support-view probabilities must be finite and non-negative")
    if bool(torch.any(groups < 0)) or bool(torch.any(groups >= int(batch_size))):
        raise ValueError("pair_group_indices contains an out-of-range value")
    if bool(torch.any(candidates < 0)) or bool(torch.any(candidates >= int(candidate_count))):
        raise ValueError("pair_candidate_indices contains an out-of-range value")
    if bool(torch.any(slots < 0)) or bool(torch.any(slots >= int(max_views))):
        raise ValueError("pair_view_slots contains an out-of-range value")

    linear = (groups * int(candidate_count) + candidates) * int(max_views) + slots
    if int(torch.unique(linear).numel()) != int(linear.numel()):
        raise ValueError("a group/candidate/view slot may be populated at most once")
    total_slots = int(batch_size) * int(candidate_count) * int(max_views)
    padded_logits = torch.index_copy(
        torch.zeros((total_slots,), device=view_logits.device, dtype=torch.float32),
        0,
        linear,
        view_logits,
    ).reshape(int(batch_size), int(candidate_count), int(max_views))
    padded_hidden = torch.index_copy(
        torch.zeros((total_slots, int(hidden.shape[1])), device=hidden.device, dtype=torch.float32),
        0,
        linear,
        hidden,
    ).reshape(int(batch_size), int(candidate_count), int(max_views), int(hidden.shape[1]))
    padded_prior = torch.index_copy(
        torch.zeros((total_slots,), device=view_logits.device, dtype=torch.float32),
        0,
        linear,
        view_prior,
    ).reshape(int(batch_size), int(candidate_count), int(max_views))
    present_flat = torch.zeros((total_slots,), device=view_logits.device, dtype=torch.bool)
    present_flat[linear] = True
    present = present_flat.reshape(int(batch_size), int(candidate_count), int(max_views))

    available_mass = torch.sum(padded_prior, dim=2)
    if bool(torch.any(available_mass > 1.0 + 2e-5)):
        raise ValueError("support-view probability mass exceeds one")
    available_mass = torch.clamp(available_mass, min=0.0, max=1.0)
    missing_mass = 1.0 - available_mass
    weighted_view_terms = torch.where(
        present & (padded_prior > 0.0),
        torch.log(padded_prior.clamp_min(1e-30)) + padded_logits,
        torch.full_like(padded_logits, -torch.inf),
    )
    missing_term = torch.where(
        missing_mass > 0.0,
        torch.log(missing_mass.clamp_min(1e-30)),
        torch.full_like(missing_mass, -torch.inf),
    )
    candidate_llr = torch.logsumexp(
        torch.cat([weighted_view_terms, missing_term[..., None]], dim=2), dim=2
    )
    candidate_hidden = torch.sum(
        padded_hidden * padded_prior[..., None], dim=2
    )
    measured = available_mass > 0.0
    coverage = available_mass
    return candidate_llr, candidate_hidden, measured, coverage


def fuse_candidate_log_likelihood_ratios(
    prior_candidate_probabilities: torch.Tensor,
    prior_unknown_probability: torch.Tensor,
    candidate_log_likelihood_ratios: torch.Tensor,
    *,
    measured_mask: torch.Tensor,
    candidate_valid: torch.Tensor | None = None,
    evidence_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse RGB evidence while preserving prior candidate-vs-unknown mass exactly."""

    prior = prior_candidate_probabilities.float()
    unknown = prior_unknown_probability.float().reshape(int(prior.shape[0]))
    llr = candidate_log_likelihood_ratios.to(device=prior.device, dtype=torch.float32)
    measured = measured_mask.to(device=prior.device, dtype=torch.bool)
    if prior.ndim != 2 or llr.shape != prior.shape or measured.shape != prior.shape:
        raise ValueError("candidate prior, RGB evidence, and measured mask must share shape (B,L)")
    if candidate_valid is None:
        valid = prior > 0.0
    else:
        valid = candidate_valid.to(device=prior.device, dtype=torch.bool)
        if valid.shape != prior.shape:
            raise ValueError("candidate_valid must share candidate probability shape")
    if bool(torch.any(prior < -1e-7)) or bool(torch.any(unknown < -1e-7)):
        raise ValueError("probabilities must be non-negative")
    mass = torch.sum(torch.where(valid, prior, torch.zeros_like(prior)), dim=1)
    total = mass + unknown
    if not bool(torch.allclose(total, torch.ones_like(total), atol=2e-5, rtol=0.0)):
        raise ValueError("candidate plus unknown probability mass must equal one")

    safe_prior = torch.where(valid, prior.clamp_min(1e-30), torch.zeros_like(prior))
    conditional = safe_prior / mass[:, None].clamp_min(1e-30)
    neutral_llr = torch.where(measured & valid, llr, torch.zeros_like(llr))
    fused_logits = torch.log(conditional.clamp_min(1e-30)) + float(evidence_weight) * neutral_llr
    fused_conditional = _masked_softmax(fused_logits, valid)
    fused_candidate = mass[:, None] * fused_conditional
    fused_candidate = torch.where(valid, fused_candidate, torch.zeros_like(fused_candidate))
    no_candidate_mass = mass <= 1e-30
    fused_candidate = torch.where(no_candidate_mass[:, None], torch.zeros_like(fused_candidate), fused_candidate)
    if not bool(torch.allclose(torch.sum(fused_candidate, dim=1), mass, atol=2e-6, rtol=0.0)):
        raise RuntimeError("RGB fusion changed candidate availability mass")
    return fused_candidate, unknown


def measurement_mode_residual_and_success(
    spatial_logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    target_offsets_xy: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    success_threshold_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the predicted local mode against target-only pose projection."""

    if spatial_logits.ndim != 2:
        raise ValueError("spatial_logits must have shape (N,K)")
    offsets = offsets_xy.to(
        device=spatial_logits.device, dtype=torch.float32
    ).reshape(-1, 2)
    target = target_offsets_xy.to(
        device=spatial_logits.device, dtype=torch.float32
    ).reshape(-1, 2)
    valid = target_valid.to(
        device=spatial_logits.device, dtype=torch.bool
    ).reshape(-1)
    if (
        int(spatial_logits.shape[0]) != int(target.shape[0])
        or int(spatial_logits.shape[1]) != int(offsets.shape[0])
        or int(valid.numel()) != int(target.shape[0])
    ):
        raise ValueError("spatial prediction and measurement targets are misaligned")
    if float(success_threshold_px) <= 0.0:
        raise ValueError("measurement success threshold must be positive")
    mode = offsets[torch.argmax(spatial_logits.detach(), dim=1)]
    residual = torch.linalg.norm(mode - target, dim=1)
    residual = torch.where(
        valid,
        residual,
        torch.full_like(residual, torch.inf),
    )
    return residual, valid & (residual <= float(success_threshold_px))


class IndependentRGBCandidateVerifier(nn.Module):
    """Rank top-L candidate identities using only query/support RGB patches."""

    def __init__(
        self,
        *,
        search_radius_px: float,
        context_radius_px: float,
        step_px: float,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        input_mode: str = "rgb_graygrad",
        encoder_arch: str = "fpn",
        template_scale_factors: Sequence[float] = (0.75, 1.0, 1.25),
        max_views: int = 4,
        measurement_validity_semantics: str = MEASUREMENT_MODE_SUCCESS_SEMANTICS,
        pose_view_mixture_enabled: bool = False,
        pair_forward_batch_size: int = 0,
        pair_forward_gradient_checkpointing: bool = False,
        identity_context_radius_px: float | None = None,
        identity_context_step_px: float | None = None,
        identity_context_layout_grid_size: int | None = None,
    ) -> None:
        super().__init__()
        if float(search_radius_px) <= 0.0 or float(context_radius_px) < 0.0 or float(step_px) <= 0.0:
            raise ValueError("RGB verifier radii and step are invalid")
        if int(max_views) <= 0:
            raise ValueError("max_views must be positive")
        if int(pair_forward_batch_size) < 0:
            raise ValueError("pair_forward_batch_size must be non-negative")
        if (identity_context_radius_px is None) != (
            identity_context_step_px is None
        ):
            raise ValueError(
                "identity context radius and step must be provided together"
            )
        if identity_context_radius_px is not None and (
            float(identity_context_radius_px) <= 0.0
            or float(identity_context_step_px) <= 0.0
        ):
            raise ValueError("identity context radius and step must be positive")
        if identity_context_layout_grid_size is not None and (
            int(identity_context_layout_grid_size) < 2
            or int(identity_context_layout_grid_size) > 8
        ):
            raise ValueError("identity context layout grid size must be in [2, 8]")
        if identity_context_layout_grid_size is not None and identity_context_radius_px is None:
            raise ValueError("identity context layout requires broad identity context")
        self.search_radius_px = float(search_radius_px)
        self.context_radius_px = float(context_radius_px)
        self.step_px = float(step_px)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_mode = str(input_mode)
        self.encoder_arch = str(encoder_arch)
        self.template_scale_factors = tuple(float(value) for value in template_scale_factors)
        self.max_views = int(max_views)
        self.measurement_validity_semantics = str(
            measurement_validity_semantics
        )
        self.pose_view_mixture_enabled = bool(pose_view_mixture_enabled)
        self.pair_forward_batch_size = int(pair_forward_batch_size)
        self.pair_forward_gradient_checkpointing = bool(
            pair_forward_gradient_checkpointing
        )
        self.identity_context_radius_px = (
            None
            if identity_context_radius_px is None
            else float(identity_context_radius_px)
        )
        self.identity_context_step_px = (
            None
            if identity_context_step_px is None
            else float(identity_context_step_px)
        )
        self.identity_context_layout_grid_size = (
            None
            if identity_context_layout_grid_size is None
            else int(identity_context_layout_grid_size)
        )
        if self.measurement_validity_semantics not in {
            MEASUREMENT_MODE_SUCCESS_SEMANTICS,
            GT_POSE_SPATIAL_DENSITY_SEMANTICS,
        }:
            raise ValueError("unsupported measurement validity semantics")
        self.encoder = TexturePatchEncoder(
            feature_dim=int(feature_dim),
            hidden_dim=int(hidden_dim),
            input_mode=str(input_mode),
            encoder_arch=str(encoder_arch),
        )
        self.identity_context_encoder = (
            None
            if self.identity_context_radius_px is None
            else TexturePatchEncoder(
                feature_dim=int(feature_dim),
                hidden_dim=int(hidden_dim),
                input_mode=str(input_mode),
                encoder_arch=str(encoder_arch),
            )
        )
        # The layout encoder must be physically separate from v6's pooled
        # context encoder. Otherwise layout-only training would silently alter
        # the pooled v6 baseline and invalidate the scale counterfactual.
        self.identity_context_layout_encoder = (
            None
            if self.identity_context_layout_grid_size is None
            else TexturePatchEncoder(
                feature_dim=int(feature_dim),
                hidden_dim=int(hidden_dim),
                input_mode=str(input_mode),
                encoder_arch=str(encoder_arch),
            )
        )
        self.logit_scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        pair_dim = int(feature_dim) * 4
        self.view_evidence = nn.Sequential(
            nn.Linear(pair_dim + 6, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.view_identity_head = nn.Linear(int(hidden_dim), 1)
        self.view_measurement_validity_head = nn.Linear(int(hidden_dim), 1)
        self.view_pose_mixture_head = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.view_pose_mixture_head.weight)
        nn.init.zeros_(self.view_pose_mixture_head.bias)
        self.register_buffer(
            "view_identity_prior_logit", torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer(
            "measured_set_availability_prior_logit",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.set_candidate_encoder = nn.Sequential(
            nn.Linear(int(hidden_dim) + 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.measured_set_availability_head = nn.Sequential(
            nn.Linear(2 * int(hidden_dim) + 1, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        if self.identity_context_encoder is not None:
            # This branch consumes broad aligned RGB context only as an identity
            # residual.  It deliberately has no offset/dustbin head: local
            # measurement density remains owned by the fine-support branch.
            self.identity_context_evidence = nn.Sequential(
                nn.Linear(pair_dim + 6, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
            )
            self.identity_context_residual = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            nn.init.zeros_(self.identity_context_residual[-1].weight)
            nn.init.zeros_(self.identity_context_residual[-1].bias)
            if self.identity_context_layout_grid_size is not None:
                assert self.identity_context_layout_encoder is not None
                # Preserve the broad crop's relative layout instead of reducing
                # it to a single pooled texture vector.  The only positional
                # channels are coordinates relative to the candidate anchor,
                # so the branch cannot use image IDs, retrieval rank, or pose.
                layout_channels = 4 * int(feature_dim) + 2
                self.identity_context_layout_fusion = nn.Sequential(
                    nn.Conv2d(layout_channels, int(hidden_dim), 1),
                    nn.GroupNorm(1, int(hidden_dim)),
                    nn.GELU(),
                    nn.Conv2d(int(hidden_dim), int(hidden_dim), 3, padding=1),
                    nn.GroupNorm(1, int(hidden_dim)),
                    nn.GELU(),
                )
                layout_features = int(hidden_dim) * int(
                    self.identity_context_layout_grid_size
                ) ** 2
                self.identity_context_layout_evidence = nn.Sequential(
                    nn.Linear(layout_features, int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.GELU(),
                )
                self.identity_context_layout_residual = nn.Sequential(
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                nn.init.zeros_(self.identity_context_layout_residual[-1].weight)
                nn.init.zeros_(self.identity_context_layout_residual[-1].bias)
            else:
                self.identity_context_layout_fusion = None
                self.identity_context_layout_evidence = None
                self.identity_context_layout_residual = None
        else:
            self.identity_context_evidence = None
            self.identity_context_residual = None
            self.identity_context_layout_encoder = None
            self.identity_context_layout_fusion = None
            self.identity_context_layout_evidence = None
            self.identity_context_layout_residual = None

    @property
    def crop_radius_px(self) -> float:
        return float(self.search_radius_px + self.context_radius_px)

    @property
    def identity_context_enabled(self) -> bool:
        return self.identity_context_encoder is not None

    @property
    def identity_context_layout_enabled(self) -> bool:
        return (
            self.identity_context_layout_encoder is not None
            and self.identity_context_layout_fusion is not None
        )

    @property
    def identity_context_crop_radius_px(self) -> float | None:
        return self.identity_context_radius_px

    def config(self) -> dict[str, object]:
        return {
            "search_radius_px": self.search_radius_px,
            "context_radius_px": self.context_radius_px,
            "step_px": self.step_px,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "input_mode": self.input_mode,
            "encoder_arch": self.encoder_arch,
            "template_scale_factors": list(self.template_scale_factors),
            "max_views": self.max_views,
            "support_view_mixture": (
                SUPPORT_VIEW_MIXTURE_CONTRACT
            ),
            "measurement_validity_semantics": (
                self.measurement_validity_semantics
            ),
            "pose_view_mixture_enabled": self.pose_view_mixture_enabled,
            "pose_view_mixture_semantics": (
                POSE_VIEW_MIXTURE_SEMANTICS
                if self.pose_view_mixture_enabled
                else "disabled"
            ),
            "pair_forward_batch_size": self.pair_forward_batch_size,
            "pair_forward_gradient_checkpointing": (
                self.pair_forward_gradient_checkpointing
            ),
            "identity_context_enabled": self.identity_context_enabled,
            "identity_context_radius_px": self.identity_context_radius_px,
            "identity_context_step_px": self.identity_context_step_px,
            "identity_context_semantics": (
                IDENTITY_CONTEXT_SEMANTICS
                if self.identity_context_enabled
                else "disabled"
            ),
            "identity_context_layout_enabled": self.identity_context_layout_enabled,
            "identity_context_layout_grid_size": self.identity_context_layout_grid_size,
            "identity_context_layout_semantics": (
                IDENTITY_CONTEXT_LAYOUT_SEMANTICS
                if self.identity_context_layout_enabled
                else "disabled"
            ),
            "view_identity_prior_logit": float(
                self.view_identity_prior_logit.detach().cpu()
            ),
            "measured_set_availability_prior_logit": float(
                self.measured_set_availability_prior_logit.detach().cpu()
            ),
        }

    def set_view_identity_prior(
        self, positive_prior: float, *, initialize_head_bias: bool = False
    ) -> None:
        prior = float(positive_prior)
        if not 0.0 < prior < 1.0:
            raise ValueError("view identity positive prior must be in (0, 1)")
        prior_logit = math.log(prior) - math.log1p(-prior)
        with torch.no_grad():
            self.view_identity_prior_logit.fill_(float(prior_logit))
            if bool(initialize_head_bias):
                self.view_identity_head.bias.fill_(float(prior_logit))

    def set_measured_set_availability_prior(
        self, positive_prior: float, *, initialize_head_bias: bool = False
    ) -> None:
        prior = float(positive_prior)
        if not 0.0 < prior < 1.0:
            raise ValueError("measured-set availability prior must be in (0, 1)")
        prior_logit = math.log(prior) - math.log1p(-prior)
        with torch.no_grad():
            self.measured_set_availability_prior_logit.fill_(float(prior_logit))
            if bool(initialize_head_bias):
                self.measured_set_availability_head[-1].bias.fill_(float(prior_logit))

    def initialize_encoder_from_measurement_checkpoint(self, checkpoint: Path) -> dict[str, object]:
        payload = torch.load(Path(checkpoint), map_location="cpu")
        state = payload["model"] if isinstance(payload, Mapping) and "model" in payload else payload
        encoder_state = {
            str(key)[len("encoder.") :]: value
            for key, value in state.items()
            if str(key).startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError("measurement checkpoint has no encoder state")
        self.encoder.load_state_dict(encoder_state, strict=True)
        if self.identity_context_encoder is not None:
            self.identity_context_encoder.load_state_dict(encoder_state, strict=True)
        if self.identity_context_layout_encoder is not None:
            self.identity_context_layout_encoder.load_state_dict(
                encoder_state, strict=True
            )
        if "logit_scale" in state:
            with torch.no_grad():
                self.logit_scale.copy_(state["logit_scale"].reshape_as(self.logit_scale))
        return dict(payload.get("config", {}) if isinstance(payload, Mapping) else {})

    def initialize_identity_context_from_local_encoder(self) -> None:
        """Clone a calibrated local encoder into a newly added context branch."""

        if self.identity_context_encoder is None:
            return
        self.identity_context_encoder.load_state_dict(self.encoder.state_dict(), strict=True)

    def initialize_identity_context_layout_from_pooled_encoder(self) -> None:
        """Clone v6 broad features into the isolated layout branch."""

        if self.identity_context_layout_encoder is None:
            return
        if self.identity_context_encoder is None:
            raise RuntimeError("layout context requires a pooled context encoder")
        self.identity_context_layout_encoder.load_state_dict(
            self.identity_context_encoder.state_dict(), strict=True
        )

    @staticmethod
    def _aligned_context_quality_features(
        query_features: torch.Tensor,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize aligned broad context without producing an offset density."""

        if query_features.shape != support_features.shape:
            raise ValueError("identity context query/support feature maps must align")
        normalized_query = F.normalize(query_features.float(), dim=1)
        normalized_support = F.normalize(support_features.float(), dim=1)
        aligned = torch.sum(normalized_query * normalized_support, dim=1)
        flattened = aligned.flatten(1)
        center = aligned[
            :,
            int(aligned.shape[1] // 2),
            int(aligned.shape[2] // 2),
        ]
        query_pool = torch.mean(query_features.flatten(2), dim=2).float()
        support_pool = torch.mean(support_features.flatten(2), dim=2).float()
        pooled_cosine = F.cosine_similarity(query_pool, support_pool, dim=1)
        return torch.stack(
            [
                torch.mean(flattened, dim=1),
                torch.std(flattened, dim=1, unbiased=False),
                torch.amax(flattened, dim=1),
                torch.amin(flattened, dim=1),
                center,
                pooled_cosine,
            ],
            dim=1,
        )

    def _forward_pair_chunk(
        self,
        query_features: torch.Tensor,
        support_patches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score a contiguous subset of independent query-support pairs."""

        support_features = self.encoder(support_patches)
        query_pool = torch.mean(query_features.flatten(2), dim=2)
        support_pool = torch.mean(support_features.flatten(2), dim=2)
        pair = torch.cat(
            [
                query_pool,
                support_pool,
                query_pool - support_pool,
                query_pool * support_pool,
            ],
            dim=1,
        )
        scale = torch.clamp(self.logit_scale, min=1.0, max=100.0)
        spatial_logits, spatial_offsets = template_search_cost_volume_logits(
            query_features,
            support_features,
            search_radius_px=self.search_radius_px,
            context_radius_px=self.context_radius_px,
            step_px=self.step_px,
            temperature=scale,
            template_scale_factors=self.template_scale_factors,
        )
        quality = cost_volume_quality_features(spatial_logits)
        view_hidden = self.view_evidence(
            torch.cat(
                [pair, quality.to(device=pair.device, dtype=pair.dtype)], dim=1
            )
        )
        return view_hidden, spatial_logits, spatial_offsets

    def _forward_pairs(
        self,
        query_features_by_group: torch.Tensor,
        support_patches: torch.Tensor,
        groups: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bound pair-level activation memory without changing group semantics."""

        query_features = query_features_by_group[groups]
        pair_count = int(support_patches.shape[0])
        limit = int(self.pair_forward_batch_size)
        if limit <= 0 or limit >= pair_count:
            return self._forward_pair_chunk(query_features, support_patches)

        view_hidden_chunks: list[torch.Tensor] = []
        spatial_logit_chunks: list[torch.Tensor] = []
        spatial_offsets: torch.Tensor | None = None
        for start in range(0, pair_count, limit):
            stop = min(pair_count, start + limit)
            query_chunk = query_features[start:stop]
            support_chunk = support_patches[start:stop]
            if (
                self.training
                and self.pair_forward_gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                result = checkpoint(
                    self._forward_pair_chunk,
                    query_chunk,
                    support_chunk,
                    use_reentrant=False,
                )
            else:
                result = self._forward_pair_chunk(query_chunk, support_chunk)
            view_hidden, spatial_logits, chunk_spatial_offsets = result
            view_hidden_chunks.append(view_hidden)
            spatial_logit_chunks.append(spatial_logits)
            # The offset grid is fixed by the model's radius/step configuration,
            # not by the pair chunk. It remains a (K,2) tensor for all pairs.
            if spatial_offsets is None:
                spatial_offsets = chunk_spatial_offsets
        if spatial_offsets is None:
            raise ValueError("pair microbatching produced no spatial offset grid")
        return (
            torch.cat(view_hidden_chunks, dim=0),
            torch.cat(spatial_logit_chunks, dim=0),
            spatial_offsets,
        )

    def _forward_identity_context_pair_chunk(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.identity_context_encoder is None or self.identity_context_evidence is None:
            raise RuntimeError("identity context branch is disabled")
        query_pool = torch.mean(query_features.flatten(2), dim=2)
        support_pool = torch.mean(support_features.flatten(2), dim=2)
        pair = torch.cat(
            [
                query_pool,
                support_pool,
                query_pool - support_pool,
                query_pool * support_pool,
            ],
            dim=1,
        )
        quality = self._aligned_context_quality_features(
            query_features, support_features
        ).to(device=pair.device, dtype=pair.dtype)
        return self.identity_context_evidence(torch.cat([pair, quality], dim=1))

    def _forward_identity_context_pairs(
        self,
        query_features_by_group: torch.Tensor,
        support_features: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """Encode broad context in the same pair chunks as the local branch."""

        query_features = query_features_by_group[groups]
        pair_count = int(support_features.shape[0])
        limit = int(self.pair_forward_batch_size)
        if limit <= 0 or limit >= pair_count:
            return self._forward_identity_context_pair_chunk(
                query_features, support_features
            )
        chunks: list[torch.Tensor] = []
        for start in range(0, pair_count, limit):
            stop = min(pair_count, start + limit)
            query_chunk = query_features[start:stop]
            support_chunk = support_features[start:stop]
            if (
                self.training
                and self.pair_forward_gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                chunk = checkpoint(
                    self._forward_identity_context_pair_chunk,
                    query_chunk,
                    support_chunk,
                    use_reentrant=False,
                )
            else:
                chunk = self._forward_identity_context_pair_chunk(
                    query_chunk, support_chunk
                )
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def _forward_identity_context_patch_chunk(
        self,
        query_features: torch.Tensor,
        support_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Encode and score one pooled-context support microbatch."""

        if self.identity_context_encoder is None:
            raise RuntimeError("identity context branch is disabled")
        support_features = self.identity_context_encoder(support_patches)
        return self._forward_identity_context_pair_chunk(
            query_features, support_features
        )

    def _forward_identity_context_layout_patch_chunk(
        self,
        query_features: torch.Tensor,
        support_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Encode and score one layout-context support microbatch."""

        if self.identity_context_layout_encoder is None:
            raise RuntimeError("identity context layout branch is disabled")
        support_features = self.identity_context_layout_encoder(support_patches)
        return self._forward_identity_context_layout_pair_chunk(
            query_features, support_features
        )

    def _forward_identity_context_patch_pairs(
        self,
        query_features_by_group: torch.Tensor,
        support_patches: torch.Tensor,
        groups: torch.Tensor,
        *,
        layout: bool,
    ) -> torch.Tensor:
        """Stream broad support encoding under the pair microbatch contract.

        Encoding every broad support crop before pair chunking defeats the
        memory bound for the FPN activations. GroupNorm makes this chunking
        batch-independent, so the operation preserves the per-pair model
        semantics while releasing support-encoder activations chunk by chunk.
        """

        query_features = query_features_by_group[groups]
        pair_count = int(support_patches.shape[0])
        limit = int(self.pair_forward_batch_size)
        forward_chunk = (
            self._forward_identity_context_layout_patch_chunk
            if bool(layout)
            else self._forward_identity_context_patch_chunk
        )
        if limit <= 0 or limit >= pair_count:
            return forward_chunk(query_features, support_patches)
        chunks: list[torch.Tensor] = []
        for start in range(0, pair_count, limit):
            stop = min(pair_count, start + limit)
            query_chunk = query_features[start:stop]
            support_chunk = support_patches[start:stop]
            if (
                self.training
                and self.pair_forward_gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                chunk = checkpoint(
                    forward_chunk,
                    query_chunk,
                    support_chunk,
                    use_reentrant=False,
                )
            else:
                chunk = forward_chunk(query_chunk, support_chunk)
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def _forward_identity_context_layout_pair_chunk(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode anchor-relative broad layout as identity evidence only."""

        if (
            self.identity_context_layout_fusion is None
            or self.identity_context_layout_evidence is None
        ):
            raise RuntimeError("identity context layout branch is disabled")
        if query_features.shape != support_features.shape:
            raise ValueError("identity context layout query/support maps must align")
        batch_size, _channels, height, width = query_features.shape
        query_normalized = F.normalize(query_features.float(), dim=1)
        support_normalized = F.normalize(support_features.float(), dim=1)
        y = torch.linspace(
            -1.0,
            1.0,
            int(height),
            device=query_features.device,
            dtype=query_features.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            int(width),
            device=query_features.device,
            dtype=query_features.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy], dim=0).expand(
            int(batch_size), -1, -1, -1
        )
        layout_input = torch.cat(
            [
                query_normalized,
                support_normalized,
                query_normalized - support_normalized,
                query_normalized * support_normalized,
                coordinates,
            ],
            dim=1,
        ).to(dtype=query_features.dtype)
        fused = self.identity_context_layout_fusion(layout_input)
        assert self.identity_context_layout_grid_size is not None
        pooled = F.adaptive_avg_pool2d(
            fused, (self.identity_context_layout_grid_size,) * 2
        ).flatten(1)
        return self.identity_context_layout_evidence(pooled)

    def _forward_identity_context_layout_pairs(
        self,
        query_features_by_group: torch.Tensor,
        support_features: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the layout branch with the same pair microbatch contract."""

        query_features = query_features_by_group[groups]
        pair_count = int(support_features.shape[0])
        limit = int(self.pair_forward_batch_size)
        if limit <= 0 or limit >= pair_count:
            return self._forward_identity_context_layout_pair_chunk(
                query_features, support_features
            )
        chunks: list[torch.Tensor] = []
        for start in range(0, pair_count, limit):
            stop = min(pair_count, start + limit)
            query_chunk = query_features[start:stop]
            support_chunk = support_features[start:stop]
            if (
                self.training
                and self.pair_forward_gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                chunk = checkpoint(
                    self._forward_identity_context_layout_pair_chunk,
                    query_chunk,
                    support_chunk,
                    use_reentrant=False,
                )
            else:
                chunk = self._forward_identity_context_layout_pair_chunk(
                    query_chunk, support_chunk
                )
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        *,
        query_patches_by_group: torch.Tensor,
        support_patches: torch.Tensor,
        pair_group_indices: torch.Tensor,
        pair_candidate_indices: torch.Tensor,
        pair_view_slots: torch.Tensor,
        pair_view_probabilities: torch.Tensor,
        candidate_valid: torch.Tensor,
        identity_context_query_patches_by_group: torch.Tensor | None = None,
        identity_context_support_patches: torch.Tensor | None = None,
        identity_context_residual_scale: float = 1.0,
        identity_context_layout_residual_scale: float = 0.0,
    ) -> RGBCandidateIdentityPrediction:
        if query_patches_by_group.ndim != 4 or support_patches.ndim != 4:
            raise ValueError("query and support patches must have shape (B,3,H,W)")
        batch_size = int(query_patches_by_group.shape[0])
        valid = candidate_valid.to(device=query_patches_by_group.device, dtype=torch.bool)
        if valid.ndim != 2 or int(valid.shape[0]) != batch_size:
            raise ValueError("candidate_valid must have shape (B,L)")
        candidate_count = int(valid.shape[1])
        groups = pair_group_indices.to(device=query_patches_by_group.device, dtype=torch.long).reshape(-1)
        if int(support_patches.shape[0]) != int(groups.numel()):
            raise ValueError("support patches and pair indices must share row count")
        if int(groups.numel()) == 0:
            raise ValueError("at least one measured RGB pair is required")
        context_residual_scale = float(identity_context_residual_scale)
        if not math.isfinite(context_residual_scale) or not (
            0.0 <= context_residual_scale <= 1.0
        ):
            raise ValueError(
                "identity_context_residual_scale must be finite and in [0, 1]"
            )
        context_layout_residual_scale = float(identity_context_layout_residual_scale)
        if not math.isfinite(context_layout_residual_scale) or not (
            0.0 <= context_layout_residual_scale <= 1.0
        ):
            raise ValueError(
                "identity_context_layout_residual_scale must be finite and in [0, 1]"
            )
        if context_layout_residual_scale > 0.0 and not self.identity_context_layout_enabled:
            raise ValueError("identity context layout residual is disabled")

        query_features_by_group = self.encoder(query_patches_by_group)
        local_view_hidden, spatial_logits, spatial_offsets = self._forward_pairs(
            query_features_by_group,
            support_patches,
            groups,
        )
        context_requested = (
            context_residual_scale > 0.0
            or context_layout_residual_scale > 0.0
        )
        if self.identity_context_enabled and context_requested:
            if (
                identity_context_query_patches_by_group is None
                or identity_context_support_patches is None
            ):
                raise ValueError(
                    "identity context patches are required by the dual-scale verifier"
                )
            context_query = identity_context_query_patches_by_group.to(
                device=query_patches_by_group.device,
                dtype=query_patches_by_group.dtype,
            )
            context_support = identity_context_support_patches.to(
                device=query_patches_by_group.device,
                dtype=support_patches.dtype,
            )
            if (
                context_query.ndim != 4
                or context_support.ndim != 4
                or int(context_query.shape[0]) != batch_size
                or int(context_support.shape[0]) != int(groups.numel())
            ):
                raise ValueError("identity context patches are misaligned with RGB pairs")
            view_hidden = local_view_hidden
            if context_residual_scale > 0.0:
                assert self.identity_context_encoder is not None
                assert self.identity_context_residual is not None
                context_features_by_group = self.identity_context_encoder(
                    context_query
                )
                context_hidden = self._forward_identity_context_patch_pairs(
                    context_features_by_group,
                    context_support,
                    groups,
                    layout=False,
                )
                view_hidden = view_hidden + context_residual_scale * (
                    self.identity_context_residual(context_hidden)
                )
            if context_layout_residual_scale > 0.0:
                assert self.identity_context_layout_encoder is not None
                assert self.identity_context_layout_residual is not None
                layout_features_by_group = self.identity_context_layout_encoder(
                    context_query
                )
                layout_hidden = self._forward_identity_context_patch_pairs(
                    layout_features_by_group,
                    context_support,
                    groups,
                    layout=True,
                )
                view_hidden = view_hidden + context_layout_residual_scale * (
                    self.identity_context_layout_residual(layout_hidden)
                )
        elif self.identity_context_enabled:
            # Evaluation-only counterfactual: keep exactly the trained local
            # branch and remove only the broad identity residual.  This avoids
            # attributing a local fine-tuning effect to broad RGB context.
            view_hidden = local_view_hidden
        else:
            if (
                identity_context_query_patches_by_group is not None
                or identity_context_support_patches is not None
            ):
                raise ValueError("identity context patches were provided to a local-only verifier")
            view_hidden = local_view_hidden
        # Keep the fine-support branch as the sole source of availability
        # evidence.  Broad context is candidate-specific identity evidence: it
        # may change which track is favored, but must not relabel a query as
        # globally available merely through the residual pathway.
        local_view_identity_logits = self.view_identity_head(
            local_view_hidden
        ).reshape(-1).float()
        view_identity_logits = self.view_identity_head(view_hidden).reshape(-1).float()
        view_measurement_validity_logits = self.view_measurement_validity_head(
            local_view_hidden
        ).reshape(-1).float()
        view_pose_mixture_logits = self.view_pose_mixture_head(
            local_view_hidden
        ).reshape(-1).float()
        (
            view_pose_mixture_probabilities,
            candidate_pose_view_probabilities,
        ) = normalize_pose_view_mixture_logits(
            view_pose_mixture_logits,
            pair_group_indices=groups,
            pair_candidate_indices=pair_candidate_indices,
            pair_view_slots=pair_view_slots,
            batch_size=batch_size,
            candidate_count=candidate_count,
            max_views=self.max_views,
        )
        view_llr = view_identity_logits - self.view_identity_prior_logit.float()
        candidate_llr, candidate_hidden, measured, coverage = aggregate_view_log_likelihood_ratios(
            view_llr,
            view_hidden,
            pair_view_probabilities=pair_view_probabilities,
            pair_group_indices=groups,
            pair_candidate_indices=pair_candidate_indices,
            pair_view_slots=pair_view_slots,
            batch_size=batch_size,
            candidate_count=candidate_count,
            max_views=self.max_views,
        )
        candidate_llr = torch.where(valid, candidate_llr, torch.zeros_like(candidate_llr))
        measured = measured & valid
        coverage = torch.where(valid, coverage, torch.zeros_like(coverage))
        conditional = _masked_softmax(candidate_llr, valid)

        if self.identity_context_enabled:
            local_view_llr = (
                local_view_identity_logits - self.view_identity_prior_logit.float()
            )
            (
                availability_candidate_llr,
                availability_candidate_hidden,
                _availability_measured,
                _availability_coverage,
            ) = aggregate_view_log_likelihood_ratios(
                local_view_llr,
                local_view_hidden,
                pair_view_probabilities=pair_view_probabilities,
                pair_group_indices=groups,
                pair_candidate_indices=pair_candidate_indices,
                pair_view_slots=pair_view_slots,
                batch_size=batch_size,
                candidate_count=candidate_count,
                max_views=self.max_views,
            )
            availability_candidate_llr = torch.where(
                valid,
                availability_candidate_llr,
                torch.zeros_like(availability_candidate_llr),
            )
            availability_candidate_hidden = torch.where(
                valid[..., None],
                availability_candidate_hidden,
                torch.zeros_like(availability_candidate_hidden),
            )
        else:
            availability_candidate_llr = candidate_llr
            availability_candidate_hidden = candidate_hidden

        candidate_set_features = self.set_candidate_encoder(
            torch.cat(
                [
                    availability_candidate_hidden,
                    availability_candidate_llr[..., None],
                    coverage[..., None],
                ],
                dim=2,
            )
        )
        candidate_set_features = torch.where(
            valid[..., None], candidate_set_features, torch.zeros_like(candidate_set_features)
        )
        valid_count = torch.sum(valid, dim=1, keepdim=True).clamp_min(1).to(candidate_set_features.dtype)
        pooled_mean = torch.sum(candidate_set_features, dim=1) / valid_count
        pooled_max = torch.max(
            torch.where(
                valid[..., None],
                candidate_set_features,
                torch.full_like(candidate_set_features, -1e4),
            ),
            dim=1,
        ).values
        any_valid = torch.any(valid, dim=1, keepdim=True)
        pooled_max = torch.where(any_valid, pooled_max, torch.zeros_like(pooled_max))
        measured_fraction = torch.sum(measured, dim=1, keepdim=True).to(candidate_set_features.dtype) / valid_count
        q_logit = self.measured_set_availability_head(
            torch.cat([pooled_mean, pooled_max, measured_fraction], dim=1)
        ).reshape(batch_size).float()
        return RGBCandidateIdentityPrediction(
            view_identity_logits=view_identity_logits,
            view_log_likelihood_ratios=view_llr,
            view_measurement_validity_logits=view_measurement_validity_logits,
            view_measurement_validity_probabilities=torch.sigmoid(
                view_measurement_validity_logits
            ),
            view_spatial_logits=spatial_logits.float(),
            view_spatial_offsets_xy=spatial_offsets,
            view_pose_mixture_logits=view_pose_mixture_logits,
            view_pose_mixture_probabilities=(
                view_pose_mixture_probabilities
            ),
            candidate_pose_view_probabilities=(
                candidate_pose_view_probabilities
            ),
            candidate_log_likelihood_ratios=candidate_llr,
            candidate_conditional_probabilities=conditional,
            candidate_valid=valid,
            candidate_measured=measured,
            candidate_view_coverage=coverage,
            measured_set_availability_logit=q_logit,
            measured_set_availability_probability=torch.sigmoid(q_logit),
        )
